/**
 * Weight Loading for PaddleOCR-VL
 *
 * Handles loading weights from HuggingFace SafeTensors format
 * with key transformations specific to PaddleOCR-VL.
 */
use crate::array::MxArray;
use napi_derive::napi;
use std::collections::HashMap;

/// Keys to ignore when loading weights
const KEYS_TO_IGNORE: &[&str] = &["packing_position_embedding", "vision_model.head"];

/// Transform weight keys from HuggingFace format to our format
pub fn transform_key(key: &str) -> String {
    let mut result = key.to_string();

    // Vision model transformations
    if result.contains("visual.vision_model") {
        if result.contains("embeddings") || result.contains("post_layernorm") {
            result = result.replace("visual.vision_model", "visual");
        } else if result.contains("encoder") {
            result = result.replace("visual.vision_model.encoder", "visual");
        }
    }

    // Projector transformation
    if result.contains("mlp_AR") {
        result = result.replace("mlp_AR", "visual.projector");
    }

    // Language model transformations
    if result.contains("model.") && !result.contains("visual") {
        result = result.replace("model.", "language_model.model.");
    }

    if result.contains("lm_head") && !result.contains("language_model") {
        result = result.replace("lm_head", "language_model.lm_head");
    }

    result
}

/// Check if a key should be ignored
pub fn should_ignore_key(key: &str) -> bool {
    for ignore in KEYS_TO_IGNORE {
        if key.contains(ignore) {
            return true;
        }
    }

    false
}

/// Check if Conv2d weight is already in MLX format.
///
/// Returns true if array is in MLX format [out_channels, kH, kW, in_channels].
/// Returns false if array needs transposition from PyTorch format [out_channels, in_channels, kH, kW].
fn is_mlx_conv_format(shape: &[i64]) -> bool {
    if shape.len() != 4 {
        return false;
    }

    let out_channels = shape[0];
    let k_h = shape[1];
    let k_w = shape[2];
    let t = shape[3];

    // If last dim is 3 (RGB), already in MLX format
    if t == 3 {
        return true;
    }

    // If out_channels is largest and k_h == k_w, already in MLX format
    if out_channels >= k_h && out_channels >= k_w && k_h == k_w {
        return true;
    }

    false
}

/// Load and sanitize PaddleOCR-VL weights
///
/// # Arguments
/// * `weights` - Raw weights from SafeTensors
///
/// # Returns
/// * Sanitized weights with transformed keys
pub fn load_paddleocr_vl_weights(
    weights: HashMap<String, MxArray>,
) -> napi::Result<HashMap<String, MxArray>> {
    let mut new_weights: HashMap<String, MxArray> = HashMap::new();
    let mut pending_qkv: HashMap<String, (Option<MxArray>, Option<MxArray>, Option<MxArray>)> =
        HashMap::new();

    for (key, value) in weights {
        // Skip ignored keys
        if should_ignore_key(&key) {
            continue;
        }

        // Handle Q/K/V merging for visual attention
        if key.contains("visual") && key.contains("self_attn") {
            // Extract base key like "visual.vision_model.encoder.layers.0.self_attn" and suffix like ".weight" or ".bias"
            if key.contains("q_proj") || key.contains("k_proj") || key.contains("v_proj") {
                // Find the proj type and suffix
                let (proj_type, base_key, suffix) = if key.contains("q_proj.weight") {
                    ("q", key.replace(".q_proj.weight", ""), ".weight")
                } else if key.contains("k_proj.weight") {
                    ("k", key.replace(".k_proj.weight", ""), ".weight")
                } else if key.contains("v_proj.weight") {
                    ("v", key.replace(".v_proj.weight", ""), ".weight")
                } else if key.contains("q_proj.bias") {
                    ("q", key.replace(".q_proj.bias", ""), ".bias")
                } else if key.contains("k_proj.bias") {
                    ("k", key.replace(".k_proj.bias", ""), ".bias")
                } else if key.contains("v_proj.bias") {
                    ("v", key.replace(".v_proj.bias", ""), ".bias")
                } else {
                    // Skip if doesn't match expected patterns
                    continue;
                };

                // Use base_key + suffix as the merge key
                let merge_key = format!("{}{}", base_key, suffix);
                let entry = pending_qkv.entry(merge_key).or_insert((None, None, None));
                match proj_type {
                    "q" => entry.0 = Some(value),
                    "k" => entry.1 = Some(value),
                    "v" => entry.2 = Some(value),
                    _ => {}
                }
                continue;
            }
        }

        // Transform the key
        let new_key = transform_key(&key);

        // Handle conv2d weight transposition
        if key.contains("patch_embedding.weight") {
            let shape = value.shape()?;
            if !is_mlx_conv_format(&shape) {
                // Transpose from [O, I, H, W] to [O, H, W, I]
                let transposed = value.transpose(Some(&[0, 2, 3, 1]))?;
                new_weights.insert(new_key, transposed);
                continue;
            }
        }

        new_weights.insert(new_key, value);
    }

    // Merge Q/K/V weights for visual attention
    for (merge_key, (q, k, v)) in pending_qkv {
        // Check which components are present before moving values
        let (has_q, has_k, has_v) = (q.is_some(), k.is_some(), v.is_some());

        if let (Some(q_weight), Some(k_weight), Some(v_weight)) = (q, k, v) {
            // Concatenate Q, K, V along axis 0
            let merged = MxArray::concatenate_many(vec![&q_weight, &k_weight, &v_weight], Some(0))?;

            // merge_key is like "visual.vision_model.encoder.layers.0.self_attn.weight"
            // We need to transform it and insert "qkv" before the suffix
            let transformed = transform_key(&merge_key);
            // transformed is like "visual.layers.0.self_attn.weight"
            // Insert "qkv" before the suffix
            let final_key = if transformed.ends_with(".weight") {
                transformed.replace(".weight", ".qkv.weight")
            } else if transformed.ends_with(".bias") {
                transformed.replace(".bias", ".qkv.bias")
            } else {
                format!("{}.qkv", transformed)
            };

            new_weights.insert(final_key, merged);
        } else {
            // Build list of missing projections
            let mut missing = Vec::new();
            if !has_q {
                missing.push("q_proj");
            }
            if !has_k {
                missing.push("k_proj");
            }
            if !has_v {
                missing.push("v_proj");
            }

            return Err(napi::Error::from_reason(format!(
                "Incomplete Q/K/V triplet for '{}': missing {}. \
                PaddleOCR-VL requires all three projections (q_proj, k_proj, v_proj) \
                for visual attention layers. Please check that model weights are complete.",
                merge_key,
                missing.join(", ")
            )));
        }
    }

    Ok(new_weights)
}

/// Get expected weight keys for PaddleOCR-VL model
#[napi]
pub fn get_expected_weight_keys() -> Vec<String> {
    let mut keys = Vec::new();

    // Vision embeddings
    keys.push("visual.embeddings.patch_embedding.weight".to_string());
    keys.push("visual.embeddings.patch_embedding.bias".to_string());
    keys.push("visual.embeddings.position_embedding.weight".to_string());

    // Vision encoder layers (27 layers)
    for i in 0..27 {
        keys.push(format!("visual.layers.{}.layer_norm1.weight", i));
        keys.push(format!("visual.layers.{}.layer_norm1.bias", i));
        keys.push(format!("visual.layers.{}.layer_norm2.weight", i));
        keys.push(format!("visual.layers.{}.layer_norm2.bias", i));
        keys.push(format!("visual.layers.{}.self_attn.qkv.weight", i));
        keys.push(format!("visual.layers.{}.self_attn.qkv.bias", i));
        keys.push(format!("visual.layers.{}.self_attn.out_proj.weight", i));
        keys.push(format!("visual.layers.{}.self_attn.out_proj.bias", i));
        keys.push(format!("visual.layers.{}.mlp.fc1.weight", i));
        keys.push(format!("visual.layers.{}.mlp.fc1.bias", i));
        keys.push(format!("visual.layers.{}.mlp.fc2.weight", i));
        keys.push(format!("visual.layers.{}.mlp.fc2.bias", i));
    }

    // Vision post-norm and projector
    keys.push("visual.post_layernorm.weight".to_string());
    keys.push("visual.post_layernorm.bias".to_string());
    keys.push("visual.projector.pre_norm.weight".to_string());
    keys.push("visual.projector.pre_norm.bias".to_string());
    keys.push("visual.projector.linear_1.weight".to_string());
    keys.push("visual.projector.linear_1.bias".to_string());
    keys.push("visual.projector.linear_2.weight".to_string());
    keys.push("visual.projector.linear_2.bias".to_string());

    // Language model
    keys.push("language_model.model.embed_tokens.weight".to_string());

    // Language model layers (18 layers)
    for i in 0..18 {
        keys.push(format!(
            "language_model.model.layers.{}.input_layernorm.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.post_attention_layernorm.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.self_attn.q_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.self_attn.k_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.self_attn.v_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.self_attn.o_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.mlp.gate_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.mlp.up_proj.weight",
            i
        ));
        keys.push(format!(
            "language_model.model.layers.{}.mlp.down_proj.weight",
            i
        ));
    }

    keys.push("language_model.model.norm.weight".to_string());
    keys.push("language_model.lm_head.weight".to_string());

    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_key_visual() {
        assert_eq!(
            transform_key("visual.vision_model.embeddings.patch_embedding.weight"),
            "visual.embeddings.patch_embedding.weight"
        );
    }

    #[test]
    fn test_transform_key_encoder() {
        assert_eq!(
            transform_key("visual.vision_model.encoder.layers.0.self_attn.qkv.weight"),
            "visual.layers.0.self_attn.qkv.weight"
        );
    }

    #[test]
    fn test_transform_key_projector() {
        assert_eq!(
            transform_key("mlp_AR.linear_1.weight"),
            "visual.projector.linear_1.weight"
        );
    }

    #[test]
    fn test_transform_key_language() {
        assert_eq!(
            transform_key("model.layers.0.self_attn.q_proj.weight"),
            "language_model.model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_should_ignore_key() {
        assert!(should_ignore_key("packing_position_embedding.weight"));
        assert!(should_ignore_key("vision_model.head.weight"));
        // Q/K/V are now collected for merging, not ignored
        assert!(!should_ignore_key(
            "visual.layers.0.self_attn.k_proj.weight"
        ));
        assert!(!should_ignore_key("visual.layers.0.self_attn.qkv.weight"));
    }

    #[test]
    fn test_is_mlx_conv_format() {
        // Already in MLX format (out_ch, kH, kW, in_ch) where in_ch=3
        assert!(is_mlx_conv_format(&[1152, 14, 14, 3]));

        // PyTorch format (out_ch, in_ch, kH, kW)
        assert!(!is_mlx_conv_format(&[1152, 3, 14, 14]));
    }

    #[test]
    fn test_expected_weight_keys_not_empty() {
        let keys = get_expected_weight_keys();
        assert!(!keys.is_empty());
    }

    #[test]
    fn test_expected_weight_keys_vision_embeddings() {
        let keys = get_expected_weight_keys();
        assert!(keys.contains(&"visual.embeddings.patch_embedding.weight".to_string()));
        assert!(keys.contains(&"visual.embeddings.position_embedding.weight".to_string()));
    }

    #[test]
    fn test_expected_weight_keys_vision_post_norm() {
        let keys = get_expected_weight_keys();
        assert!(keys.contains(&"visual.post_layernorm.weight".to_string()));
        assert!(keys.contains(&"visual.post_layernorm.bias".to_string()));
    }

    #[test]
    fn test_expected_weight_keys_projector() {
        let keys = get_expected_weight_keys();
        assert!(keys.contains(&"visual.projector.pre_norm.weight".to_string()));
        assert!(keys.contains(&"visual.projector.pre_norm.bias".to_string()));
        assert!(keys.contains(&"visual.projector.linear_1.weight".to_string()));
        assert!(keys.contains(&"visual.projector.linear_1.bias".to_string()));
        assert!(keys.contains(&"visual.projector.linear_2.weight".to_string()));
        assert!(keys.contains(&"visual.projector.linear_2.bias".to_string()));
    }

    #[test]
    fn test_expected_weight_keys_language_model() {
        let keys = get_expected_weight_keys();
        assert!(keys.contains(&"language_model.model.embed_tokens.weight".to_string()));
        assert!(keys.contains(&"language_model.model.norm.weight".to_string()));
        assert!(keys.contains(&"language_model.lm_head.weight".to_string()));
    }

    #[test]
    fn test_expected_weight_keys_vision_layer_count() {
        let keys = get_expected_weight_keys();

        // Count vision layer keys (27 layers)
        let layer_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.starts_with("visual.layers."))
            .collect();

        // Each layer has 12 keys: layer_norm1 (weight, bias), layer_norm2 (weight, bias),
        // self_attn (qkv weight, qkv bias, out_proj weight, out_proj bias),
        // mlp (fc1 weight, fc1 bias, fc2 weight, fc2 bias)
        assert_eq!(layer_keys.len(), 27 * 12);
    }

    #[test]
    fn test_expected_weight_keys_language_layer_count() {
        let keys = get_expected_weight_keys();

        // Count language layer keys (18 layers)
        let layer_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.starts_with("language_model.model.layers."))
            .collect();

        // Each layer has 9 keys: input_layernorm, post_attention_layernorm,
        // q/k/v/o_proj, gate/up/down_proj
        assert_eq!(layer_keys.len(), 18 * 9);
    }

    #[test]
    fn test_expected_weight_keys_specific_layers() {
        let keys = get_expected_weight_keys();

        // Check specific vision layer keys exist
        assert!(keys.contains(&"visual.layers.0.layer_norm1.weight".to_string()));
        assert!(keys.contains(&"visual.layers.26.self_attn.qkv.weight".to_string()));

        // Check specific language layer keys exist
        assert!(
            keys.contains(&"language_model.model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert!(keys.contains(&"language_model.model.layers.17.mlp.down_proj.weight".to_string()));
    }
}
