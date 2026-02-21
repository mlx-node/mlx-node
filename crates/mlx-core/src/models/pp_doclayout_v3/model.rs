//! PP-DocLayoutV3 Full Model
//!
//! Top-level model combining backbone, encoder, decoder, and heads.
//! Provides the NAPI `DocLayoutModel` class with `load()` and `detect()` methods.
//!
//! Architecture (inference path):
//! 1. Image preprocessing: resize to 800x800, rescale to [0,1]
//! 2. Backbone (HGNetV2-L): extract 4 feature maps at strides 4, 8, 16, 32
//! 3. Encoder input projection: project 3 feature maps (strides 8, 16, 32) to hidden_dim
//! 4. Hybrid Encoder: AIFI transformer + FPN + PAN + mask feature head
//! 5. Decoder input projection: project PAN features
//! 6. Generate anchors + select top-K queries from encoder output
//! 7. Mask-enhanced reference point refinement
//! 8. Decoder: iterative bbox refinement + class prediction + reading order
//! 9. Post-processing: score thresholding + box decoding + reading order

use crate::array::MxArray;
use crate::nn::linear::Linear;
use crate::nn::normalization::LayerNorm;
use napi::Either;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::backbone::HGNetV2Backbone;
use super::config::PPDocLayoutV3Config;
use super::decoder::{Decoder, inverse_sigmoid, mask_to_box_coordinate};
use super::encoder::{HybridEncoder, InputProjection};
use super::heads::{GlobalPointer, MLPPredictionHead};
use super::postprocessing::{LayoutElement, postprocess_detection};
use super::processing::PPDocLayoutV3ImageProcessor;

/// PP-DocLayoutV3 full model for document layout analysis.
///
/// Combines HGNetV2 backbone, hybrid encoder, and RT-DETR decoder
/// with mask-enhanced attention and reading order prediction.
///
/// Weights must be downloaded from `PaddlePaddle/PP-DocLayoutV3_safetensors` on HuggingFace.
/// The regular `PaddlePaddle/PP-DocLayoutV3` repo uses PaddlePaddle format and is not compatible.
#[napi(js_name = "DocLayoutModel")]
pub struct PPDocLayoutV3Model {
    config: PPDocLayoutV3Config,
    backbone: HGNetV2Backbone,
    encoder_input_proj: Vec<InputProjection>,
    encoder: HybridEncoder,
    enc_output_linear: Linear,
    enc_output_norm: LayerNorm,
    enc_score_head: Linear,
    enc_bbox_head: MLPPredictionHead,
    decoder_input_proj: Vec<InputProjection>,
    decoder: Decoder,
    decoder_order_head: Vec<Linear>,
    decoder_global_pointer: GlobalPointer,
    decoder_norm: LayerNorm,
    mask_query_head: MLPPredictionHead,
    _denoising_class_embed: Option<MxArray>,
    image_processor: PPDocLayoutV3ImageProcessor,
}

/// Generate anchor points for all feature levels.
///
/// For each level with the given spatial shapes, generates a grid of anchor
/// points in normalized coordinates with associated width/height.
///
/// # Arguments
/// * `spatial_shapes_list` - Vec of (height, width) for each feature level
/// * `grid_size` - Base grid size (default 0.05)
///
/// # Returns
/// * `(anchors, valid_mask)` where:
///   - anchors: [1, total_tokens, 4] in inverse_sigmoid space
///   - valid_mask: [1, total_tokens, 1] boolean mask for valid anchors
fn generate_anchors(
    spatial_shapes_list: &[(i64, i64)],
    grid_size: f64,
) -> Result<(MxArray, MxArray)> {
    let mut all_anchors: Vec<MxArray> = Vec::new();

    for (level, &(height, width)) in spatial_shapes_list.iter().enumerate() {
        // Create grid
        let grid_y = MxArray::arange(0.0, height as f64, Some(1.0), None)?;
        let grid_x = MxArray::arange(0.0, width as f64, Some(1.0), None)?;

        // Meshgrid: grid_y varies along rows, grid_x along columns
        let grid_x_2d = grid_x
            .reshape(&[1, width])?
            .broadcast_to(&[height, width])?;
        let grid_y_2d = grid_y
            .reshape(&[height, 1])?
            .broadcast_to(&[height, width])?;

        // Stack [grid_x, grid_y] and add 0.5, normalize
        let grid_x_norm = grid_x_2d.add_scalar(0.5)?.div_scalar(width as f64)?;
        let grid_y_norm = grid_y_2d.add_scalar(0.5)?.div_scalar(height as f64)?;

        // Stack to [height, width, 2]
        let grid_xy = MxArray::stack(vec![&grid_x_norm, &grid_y_norm], Some(-1))?;

        // Width/height: ones * grid_size * 2^level
        let wh_val = grid_size * (2.0f64).powi(level as i32);
        let wh = MxArray::full(&[height, width, 2], Either::A(wh_val), None)?;

        // Concatenate [grid_xy, wh] → [height, width, 4]
        let anchors_level = MxArray::concatenate(&grid_xy, &wh, 2)?;

        // Reshape to [1, height*width, 4]
        let anchors_flat = anchors_level.reshape(&[1, height * width, 4])?;
        all_anchors.push(anchors_flat);
    }

    // Concatenate all levels: [1, total_tokens, 4]
    let anchor_refs: Vec<&MxArray> = all_anchors.iter().collect();
    let anchors = MxArray::concatenate_many(anchor_refs, Some(1))?;

    // Valid mask: all coordinates in (eps, 1-eps)
    let eps = 1e-2;
    let eps_arr = MxArray::from_float32(&[eps as f32], &[1])?;
    let one_minus_eps_arr = MxArray::from_float32(&[(1.0 - eps) as f32], &[1])?;
    let above_eps = anchors.greater(&eps_arr)?;
    let below_1_eps = anchors.less(&one_minus_eps_arr)?;
    let valid_per_coord = above_eps.logical_and(&below_1_eps)?;

    // All 4 coordinates must be valid: reduce along last dim
    // min along axis -1 for boolean (all true → true)
    let valid_mask = valid_per_coord.min(Some(&[-1]), Some(true))?;

    // Apply inverse sigmoid to anchors
    let anchors_clamped = anchors.clip(Some(eps), Some(1.0 - eps))?;
    let inv_anchors = inverse_sigmoid(&anchors_clamped, eps)?;

    // Where invalid, fill with large value
    let large_val = MxArray::full(&[1], Either::A(f32::MAX as f64 * 0.5), None)?;
    let inv_anchors = valid_mask.where_(&inv_anchors, &large_val)?;

    Ok((inv_anchors, valid_mask))
}

#[napi]
impl PPDocLayoutV3Model {
    /// Load a PP-DocLayoutV3 model from a directory containing `config.json` and `model.safetensors`.
    ///
    /// The model directory should be cloned from `PaddlePaddle/PP-DocLayoutV3_safetensors` on HuggingFace.
    ///
    /// # Arguments
    /// * `model_path` - Path to model directory
    ///
    /// # Returns
    /// * Initialized DocLayoutModel ready for inference
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        super::persistence::load_model(&model_path)
    }

    /// Detect document layout elements in an image.
    ///
    /// # Arguments
    /// * `image_data` - Encoded image bytes (PNG/JPEG)
    /// * `threshold` - Optional confidence threshold (default 0.5)
    ///
    /// # Returns
    /// * Vec of LayoutElements sorted by reading order
    #[napi]
    pub fn detect(&self, image_data: Buffer, threshold: Option<f64>) -> Result<Vec<LayoutElement>> {
        let threshold = threshold.unwrap_or(0.5);

        // 1. Preprocess image
        let (pixel_values, orig_h, orig_w) = self.image_processor.process(&image_data)?;

        // 2. Run backbone
        let features = self.backbone.forward(&pixel_values)?;
        // features: [stage1(H/4), stage2(H/8), stage3(H/16), stage4(H/32)]

        // 3. Extract x4_feat (stride-4 feature) and remaining features
        let x4_feat = &features[0]; // stage1: [B, H/4, W/4, 128]
        let encoder_features = &features[1..]; // stage2, stage3, stage4

        // 4. Apply encoder input projections
        let mut proj_feats: Vec<MxArray> = Vec::with_capacity(encoder_features.len());
        for (level, feat) in encoder_features.iter().enumerate() {
            let projected = self.encoder_input_proj[level].forward(feat)?;
            proj_feats.push(projected);
        }

        // 5. Run hybrid encoder
        let (pan_features, mask_feat) = self.encoder.forward(&mut proj_feats, x4_feat)?;

        // 6. Apply decoder input projections and flatten
        let mut source_flatten_parts: Vec<MxArray> = Vec::new();
        let mut spatial_shapes_list: Vec<(i64, i64)> = Vec::new();

        for (level, pan_feat) in pan_features.iter().enumerate() {
            let projected = self.decoder_input_proj[level].forward(pan_feat)?;
            let proj_shape = projected.shape()?;
            let proj_shape: Vec<i64> = proj_shape.as_ref().to_vec();
            let batch = proj_shape[0];
            let h = proj_shape[1];
            let w = proj_shape[2];
            let c = proj_shape[3];

            spatial_shapes_list.push((h, w));

            // Flatten spatial dims: [B, H, W, C] → [B, H*W, C]
            let flat = projected.reshape(&[batch, h * w, c])?;
            source_flatten_parts.push(flat);
        }

        let flat_refs: Vec<&MxArray> = source_flatten_parts.iter().collect();
        let source_flatten = MxArray::concatenate_many(flat_refs, Some(1))?;

        // Compute spatial_shapes tensor [num_levels, 2]
        let num_levels = spatial_shapes_list.len();
        let mut shapes_data: Vec<f32> = Vec::with_capacity(num_levels * 2);
        for &(h, w) in &spatial_shapes_list {
            shapes_data.push(h as f32);
            shapes_data.push(w as f32);
        }
        let spatial_shapes = MxArray::from_float32(&shapes_data, &[num_levels as i64, 2])?;

        // Compute level_start_index
        let mut start_indices: Vec<f32> = Vec::with_capacity(num_levels);
        let mut cumsum: f32 = 0.0;
        for &(h, w) in &spatial_shapes_list {
            start_indices.push(cumsum);
            cumsum += (h * w) as f32;
        }
        let level_start_index = MxArray::from_float32(&start_indices, &[num_levels as i64])?;

        // 7. Generate anchors
        let (anchors, valid_mask) = generate_anchors(&spatial_shapes_list, 0.05)?;

        // 8. Apply valid mask and compute encoder output
        let valid_mask_float = valid_mask.astype(crate::array::DType::Float32)?;
        let memory = valid_mask_float.mul(&source_flatten)?;

        // enc_output = enc_output_norm(enc_output_linear(memory))
        let output_memory = self.enc_output_linear.forward(&memory)?;
        let output_memory = self.enc_output_norm.forward(&output_memory)?;

        // 9. Score and bbox prediction
        let enc_outputs_class = self.enc_score_head.forward(&output_memory)?;
        let enc_outputs_coord = self.enc_bbox_head.forward(&output_memory)?;
        let enc_outputs_coord = enc_outputs_coord.add(&anchors)?;

        // 10. Top-K query selection
        let num_queries = self.config.num_queries as i64;

        // enc_outputs_class.max(-1) → [B, total_tokens]
        let class_max = enc_outputs_class.max(Some(&[-1]), Some(false))?;

        // Argsort descending along dim 1 to get top-k indices
        let neg_class_max = class_max.negative()?;
        let sorted_indices = neg_class_max.argsort(Some(1))?;

        // Take first num_queries indices
        let sf_shape = source_flatten.shape()?;
        let batch_size = sf_shape[0];

        let topk_indices = sorted_indices.slice(&[0, 0], &[batch_size, num_queries])?;

        // Gather reference points and target queries
        let d_model = self.config.d_model as i64;
        let num_labels = self.config.num_labels as i64;

        // Expand topk_indices for gathering: [B, num_queries] → [B, num_queries, 4]
        let topk_idx_box = topk_indices
            .expand_dims(2)?
            .broadcast_to(&[batch_size, num_queries, 4])?
            .astype(crate::array::DType::Int32)?;
        let reference_points_unact = enc_outputs_coord.take_along_axis(&topk_idx_box, 1)?;

        // Gather target queries from output_memory
        let topk_idx_emb = topk_indices
            .expand_dims(2)?
            .broadcast_to(&[batch_size, num_queries, d_model])?
            .astype(crate::array::DType::Int32)?;
        let target = output_memory.take_along_axis(&topk_idx_emb, 1)?;

        // 11. Mask-enhanced reference points (if enabled)
        let reference_points_unact = if self.config.mask_enhanced {
            // Compute encoder output masks from the selected queries
            let out_query = self.decoder_norm.forward(&target)?;
            let mask_query_embed = self.mask_query_head.forward(&out_query)?;

            // mask_feat needs to be in NCHW for bmm: [B, H, W, C] → [B, C, H, W]
            let mask_feat_nchw = mask_feat.transpose(Some(&[0, 3, 1, 2]))?;
            let mf_shape = mask_feat_nchw.shape()?;
            let mf_shape: Vec<i64> = mf_shape.as_ref().to_vec();
            let mask_h = mf_shape[2];
            let mask_w = mf_shape[3];
            let num_proto = mf_shape[1];

            // Flatten mask_feat spatial dims: [B, num_proto, H*W]
            let mask_feat_flat =
                mask_feat_nchw.reshape(&[batch_size, num_proto, mask_h * mask_w])?;

            // bmm: [B, num_queries, num_proto] @ [B, num_proto, H*W] → [B, num_queries, H*W]
            let enc_out_masks = mask_query_embed.matmul(&mask_feat_flat)?;
            let enc_out_masks =
                enc_out_masks.reshape(&[batch_size, num_queries, mask_h, mask_w])?;

            // mask_to_box_coordinate(enc_out_masks > 0)
            let zero = MxArray::from_float32(&[0.0], &[1])?;
            let mask_bool = enc_out_masks.greater(&zero)?;
            let reference_points = mask_to_box_coordinate(&mask_bool)?;
            inverse_sigmoid(&reference_points, 1e-5)?
        } else {
            reference_points_unact
        };

        // 12. Convert mask_feat to NCHW for decoder
        let mask_feat_nchw = mask_feat.transpose(Some(&[0, 3, 1, 2]))?;

        // 13. Run decoder
        let decoder_output = self.decoder.forward(
            &target,
            &source_flatten,
            &reference_points_unact,
            &spatial_shapes,
            &spatial_shapes_list,
            &level_start_index,
            None, // no attention mask at inference
            &self.decoder_order_head,
            &self.decoder_global_pointer,
            &self.mask_query_head,
            &self.decoder_norm,
            &mask_feat_nchw,
        )?;

        // 14. Extract last layer outputs
        let num_decoder_layers = self.config.decoder_layers as i64;

        // intermediate_logits: [B, num_layers, num_queries, num_labels]
        // Take last layer: [:, -1, :, :]
        let logits = decoder_output.intermediate_logits.slice(
            &[0, num_decoder_layers - 1, 0, 0],
            &[batch_size, num_decoder_layers, num_queries, num_labels],
        )?;
        let logits = logits.squeeze(Some(&[1]))?;

        // intermediate_reference_points: [B, num_layers, num_queries, 4]
        let pred_boxes = decoder_output.intermediate_reference_points.slice(
            &[0, num_decoder_layers - 1, 0, 0],
            &[batch_size, num_decoder_layers, num_queries, 4],
        )?;
        let pred_boxes = pred_boxes.squeeze(Some(&[1]))?;

        // decoder_out_order_logits: [B, num_layers, num_queries, num_queries]
        let order_logits = decoder_output.decoder_out_order_logits.slice(
            &[0, num_decoder_layers - 1, 0, 0],
            &[batch_size, num_decoder_layers, num_queries, num_queries],
        )?;
        let order_logits = order_logits.squeeze(Some(&[1]))?;

        // 15. Post-process
        postprocess_detection(
            &logits,
            &pred_boxes,
            &order_logits,
            orig_h,
            orig_w,
            threshold,
            &self.config.id2label,
        )
    }
}

/// Create an uninitialized PPDocLayoutV3Model (used by persistence).
///
/// This is called from persistence.rs after loading all weights.
pub(super) fn create_model(
    config: PPDocLayoutV3Config,
    backbone: HGNetV2Backbone,
    encoder_input_proj: Vec<InputProjection>,
    encoder: HybridEncoder,
    enc_output_linear: Linear,
    enc_output_norm: LayerNorm,
    enc_score_head: Linear,
    enc_bbox_head: MLPPredictionHead,
    decoder_input_proj: Vec<InputProjection>,
    decoder: Decoder,
    decoder_order_head: Vec<Linear>,
    decoder_global_pointer: GlobalPointer,
    decoder_norm: LayerNorm,
    mask_query_head: MLPPredictionHead,
    denoising_class_embed: Option<MxArray>,
) -> PPDocLayoutV3Model {
    PPDocLayoutV3Model {
        config,
        backbone,
        encoder_input_proj,
        encoder,
        enc_output_linear,
        enc_output_norm,
        enc_score_head,
        enc_bbox_head,
        decoder_input_proj,
        decoder,
        decoder_order_head,
        decoder_global_pointer,
        decoder_norm,
        mask_query_head,
        _denoising_class_embed: denoising_class_embed,
        image_processor: PPDocLayoutV3ImageProcessor::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_anchors() {
        let shapes = vec![(100, 100), (50, 50), (25, 25)];
        let (anchors, valid_mask) = generate_anchors(&shapes, 0.05).unwrap();

        let a_shape: Vec<i64> = anchors.shape().unwrap().as_ref().to_vec();
        let total = 100 * 100 + 50 * 50 + 25 * 25;
        assert_eq!(a_shape, vec![1, total, 4]);

        let v_shape: Vec<i64> = valid_mask.shape().unwrap().as_ref().to_vec();
        assert_eq!(v_shape, vec![1, total, 1]);
    }
}
