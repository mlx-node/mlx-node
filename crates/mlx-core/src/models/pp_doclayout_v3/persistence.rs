//! PP-DocLayoutV3 Weight Loading
//!
//! SafeTensors weight loading for the PP-DocLayoutV3 model.
//! Handles key mapping from HuggingFace naming conventions and
//! Conv2d weight transposition from PyTorch [O,I,kH,kW] to MLX [O,kH,kW,I].

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::persistence::load_all_safetensors;
use crate::vision::cnn::{
    load_conv_bn_act, load_conv2d, load_frozen_bn, load_hgnetv2_backbone, load_layer_norm,
    load_linear,
};

use super::config::PPDocLayoutV3Config;
use super::decoder::{Decoder, DecoderLayer, MultiscaleDeformableAttention, SelfAttention};
use super::encoder::{
    CSPRepLayer, Encoder, EncoderLayer, EncoderMaskOutput, HybridEncoder, InputProjection,
    MaskFeatFPN, MultiheadAttention, RepVggBlock, ScaleHead, ScaleHeadLayer,
};
use super::heads::{GlobalPointer, MLPPredictionHead};
use super::model::{PPDocLayoutV3Model, create_model};

// ============================================================================
// Helper functions
// ============================================================================

/// Load an InputProjection (1x1 Conv + BN) from parameters.
fn load_input_projection(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<InputProjection> {
    let conv = load_conv2d(
        params,
        &format!("{}.0", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        false,
    )?;
    let norm = load_frozen_bn(params, &format!("{}.1", prefix), eps)?;
    Ok(InputProjection::new(conv, norm))
}

/// Load an MLPPredictionHead from parameters.
fn load_mlp_head(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    num_layers: usize,
) -> Result<MLPPredictionHead> {
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let linear = load_linear(params, &format!("{}.layers.{}", prefix, i))?;
        layers.push(linear);
    }
    Ok(MLPPredictionHead::from_layers(layers))
}

// ============================================================================
// Encoder Loading
// ============================================================================

fn load_hybrid_encoder(
    params: &HashMap<String, MxArray>,
    config: &PPDocLayoutV3Config,
) -> Result<HybridEncoder> {
    let eps = config.batch_norm_eps;
    let ln_eps = config.layer_norm_eps;
    let _hidden_dim = config.encoder_hidden_dim;
    let act_fn = config.activation_function.as_str();
    let enc_act = config.encoder_activation_function.as_str();

    // Load transformer encoder(s)
    // HF: self.aifi = nn.ModuleList([PPDocLayoutV3AIFILayer(...)])
    // PPDocLayoutV3AIFILayer has self.layers = nn.ModuleList([PPDocLayoutV3EncoderLayer(...)])
    let mut encoders = Vec::new();
    for _enc_idx in 0..config.encode_proj_layers.len() {
        // Each encoder has config.encoder_layers layers
        let mut enc_layers = Vec::new();
        for layer_idx in 0..config.encoder_layers {
            let prefix = format!("encoder.encoder.{}.layers.{}", _enc_idx, layer_idx);

            // Self-attention (PPDocLayoutV3SelfAttention uses q_proj, k_proj, v_proj, o_proj)
            let q_proj = load_linear(params, &format!("{}.self_attn.q_proj", prefix))?;
            let k_proj = load_linear(params, &format!("{}.self_attn.k_proj", prefix))?;
            let v_proj = load_linear(params, &format!("{}.self_attn.v_proj", prefix))?;
            let out_proj = load_linear(params, &format!("{}.self_attn.out_proj", prefix))?;
            let self_attn = MultiheadAttention::new(
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                config.encoder_attention_heads,
            )?;

            let self_attn_ln =
                load_layer_norm(params, &format!("{}.self_attn_layer_norm", prefix), ln_eps)?;

            // FFN is inside self.mlp = PPDocLayoutV3MLP which has self.fc1, self.fc2
            let fc1 = load_linear(params, &format!("{}.fc1", prefix))?;
            let fc2 = load_linear(params, &format!("{}.fc2", prefix))?;

            let final_ln =
                load_layer_norm(params, &format!("{}.final_layer_norm", prefix), ln_eps)?;

            enc_layers.push(EncoderLayer::new(
                self_attn,
                self_attn_ln,
                fc1,
                fc2,
                final_ln,
                enc_act,
                config.normalize_before,
            ));
        }
        encoders.push(Encoder::new(enc_layers));
    }

    // Load FPN lateral convs
    let num_levels = config.encoder_in_channels.len();
    let num_fpn = num_levels - 1;

    let mut lateral_convs = Vec::with_capacity(num_fpn);
    for i in 0..num_fpn {
        let prefix = format!("encoder.lateral_convs.{}", i);
        let cnl = load_conv_bn_act(
            params,
            &format!("{}.conv", prefix),
            &format!("{}.norm", prefix),
            act_fn,
            (1, 1),
            (0, 0),
            1,
            eps,
            None,
        )?;
        lateral_convs.push(cnl);
    }

    // Load FPN CSPRepLayer blocks
    let mut fpn_blocks = Vec::with_capacity(num_fpn);
    for i in 0..num_fpn {
        let prefix = format!("encoder.fpn_blocks.{}", i);
        fpn_blocks.push(load_csp_rep_layer(params, &prefix, act_fn, eps)?);
    }

    // Load PAN downsample convs
    let mut downsample_convs = Vec::with_capacity(num_fpn);
    for i in 0..num_fpn {
        let prefix = format!("encoder.downsample_convs.{}", i);
        let cnl = load_conv_bn_act(
            params,
            &format!("{}.conv", prefix),
            &format!("{}.norm", prefix),
            act_fn,
            (2, 2),
            (1, 1),
            1,
            eps,
            None,
        )?;
        downsample_convs.push(cnl);
    }

    // Load PAN CSPRepLayer blocks
    let mut pan_blocks = Vec::with_capacity(num_fpn);
    for i in 0..num_fpn {
        let prefix = format!("encoder.pan_blocks.{}", i);
        pan_blocks.push(load_csp_rep_layer(params, &prefix, act_fn, eps)?);
    }

    // Load MaskFeatFPN
    let mask_feature_head = load_mask_feat_fpn(params, config)?;

    // Load encoder_mask_lateral
    // HF: PPDocLayoutV3ConvLayer uses .convolution and .normalization
    // This is a 3x3 conv so padding = (kernel_size - 1) / 2 = 1
    let encoder_mask_lateral = load_conv_bn_act(
        params,
        "encoder.encoder_mask_lateral.convolution",
        "encoder.encoder_mask_lateral.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
        None,
    )?;

    // Load encoder_mask_output
    // HF: PPDocLayoutV3EncoderMaskOutput.base_conv = PPDocLayoutV3ConvLayer (uses .convolution, .normalization)
    let base_conv = load_conv_bn_act(
        params,
        "encoder.encoder_mask_output.base_conv.convolution",
        "encoder.encoder_mask_output.base_conv.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
        None,
    )?;
    let mask_output_conv = load_conv2d(
        params,
        "encoder.encoder_mask_output.conv",
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
    )?;
    let encoder_mask_output = EncoderMaskOutput::new(base_conv, mask_output_conv);

    Ok(HybridEncoder::new(
        config,
        encoders,
        lateral_convs,
        fpn_blocks,
        downsample_convs,
        pan_blocks,
        mask_feature_head,
        encoder_mask_lateral,
        encoder_mask_output,
    ))
}

fn load_csp_rep_layer(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    act_fn: &str,
    eps: f64,
) -> Result<CSPRepLayer> {
    let conv1 = load_conv_bn_act(
        params,
        &format!("{}.conv1.conv", prefix),
        &format!("{}.conv1.norm", prefix),
        act_fn,
        (1, 1),
        (0, 0),
        1,
        eps,
        None,
    )?;
    let conv2 = load_conv_bn_act(
        params,
        &format!("{}.conv2.conv", prefix),
        &format!("{}.conv2.norm", prefix),
        act_fn,
        (1, 1),
        (0, 0),
        1,
        eps,
        None,
    )?;

    // Load bottleneck RepVGG blocks
    let mut bottlenecks = Vec::new();
    let mut idx = 0;
    loop {
        let key = format!("{}.bottlenecks.{}.conv1.conv.weight", prefix, idx);
        if !params.contains_key(&key) {
            break;
        }
        let rep_conv1 = load_conv_bn_act(
            params,
            &format!("{}.bottlenecks.{}.conv1.conv", prefix, idx),
            &format!("{}.bottlenecks.{}.conv1.norm", prefix, idx),
            "none",
            (1, 1),
            (1, 1),
            1,
            eps,
            None,
        )?;
        let rep_conv2 = load_conv_bn_act(
            params,
            &format!("{}.bottlenecks.{}.conv2.conv", prefix, idx),
            &format!("{}.bottlenecks.{}.conv2.norm", prefix, idx),
            "none",
            (1, 1),
            (0, 0),
            1,
            eps,
            None,
        )?;
        bottlenecks.push(RepVggBlock::new(rep_conv1, rep_conv2, act_fn));
        idx += 1;
    }

    // Optional conv3
    let conv3_key = format!("{}.conv3.conv.weight", prefix);
    let conv3 = if params.contains_key(&conv3_key) {
        Some(load_conv_bn_act(
            params,
            &format!("{}.conv3.conv", prefix),
            &format!("{}.conv3.norm", prefix),
            act_fn,
            (1, 1),
            (0, 0),
            1,
            eps,
            None,
        )?)
    } else {
        None
    };

    Ok(CSPRepLayer::new(conv1, conv2, bottlenecks, conv3))
}

fn load_mask_feat_fpn(
    params: &HashMap<String, MxArray>,
    config: &PPDocLayoutV3Config,
) -> Result<MaskFeatFPN> {
    let eps = config.batch_norm_eps;
    let act_fn = config.activation_function.as_str();
    let feat_strides = &config.feat_strides;
    let num_levels = feat_strides.len();

    // Reorder by stride ascending (they already are for default config)
    let mut stride_indices: Vec<(usize, i32)> = feat_strides
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    stride_indices.sort_by_key(|&(_, s)| s);
    let reorder_index: Vec<usize> = stride_indices.iter().map(|&(i, _)| i).collect();
    let sorted_strides: Vec<i32> = stride_indices.iter().map(|&(_, s)| s).collect();

    // Build scale heads
    let base_stride = sorted_strides[0];
    let mut scale_heads = Vec::with_capacity(num_levels);

    #[allow(clippy::needless_range_loop)]
    for i in 0..num_levels {
        let fpn_stride = sorted_strides[i];
        let num_scale_layers = (fpn_stride as f64 / base_stride as f64).log2() as usize;
        let mut layers = Vec::new();

        if num_scale_layers == 0 {
            // Single conv, no upsample
            // HF: PPDocLayoutV3ScaleHead.layers[0] = PPDocLayoutV3ConvLayer (uses .convolution, .normalization)
            let conv = load_conv_bn_act(
                params,
                &format!(
                    "encoder.mask_feature_head.scale_heads.{}.layers.0.convolution",
                    i
                ),
                &format!(
                    "encoder.mask_feature_head.scale_heads.{}.layers.0.normalization",
                    i
                ),
                act_fn,
                (1, 1),
                (1, 1),
                1,
                eps,
                None,
            )?;
            layers.push(ScaleHeadLayer::Conv(conv));
        } else {
            for j in 0..num_scale_layers {
                // HF: layers[j*2] = PPDocLayoutV3ConvLayer, layers[j*2+1] = nn.Upsample
                let conv = load_conv_bn_act(
                    params,
                    &format!(
                        "encoder.mask_feature_head.scale_heads.{}.layers.{}.convolution",
                        i,
                        j * 2
                    ),
                    &format!(
                        "encoder.mask_feature_head.scale_heads.{}.layers.{}.normalization",
                        i,
                        j * 2
                    ),
                    act_fn,
                    (1, 1),
                    (1, 1),
                    1,
                    eps,
                    None,
                )?;
                layers.push(ScaleHeadLayer::Conv(conv));
                layers.push(ScaleHeadLayer::Upsample);
            }
        }

        scale_heads.push(ScaleHead::new(layers));
    }

    // Output conv
    // HF: PPDocLayoutV3ConvLayer uses .convolution and .normalization
    let output_conv = load_conv_bn_act(
        params,
        "encoder.mask_feature_head.output_conv.convolution",
        "encoder.mask_feature_head.output_conv.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
        None,
    )?;

    Ok(MaskFeatFPN::new(reorder_index, scale_heads, output_conv))
}

// ============================================================================
// Decoder Loading
// ============================================================================

fn load_decoder(
    params: &HashMap<String, MxArray>,
    config: &PPDocLayoutV3Config,
) -> Result<Decoder> {
    let d_model = config.d_model;
    let num_heads = config.decoder_attention_heads;
    let n_levels = config.num_feature_levels;
    let n_points = config.decoder_n_points;
    let ln_eps = config.layer_norm_eps;
    let dec_act = config.decoder_activation_function.as_str();

    let mut decoder_layers = Vec::with_capacity(config.decoder_layers as usize);
    for i in 0..config.decoder_layers {
        let prefix = format!("decoder.layers.{}", i);

        // Self-attention (PPDocLayoutV3SelfAttention uses q_proj, k_proj, v_proj, o_proj)
        let sa_q = load_linear(params, &format!("{}.self_attn.q_proj", prefix))?;
        let sa_k = load_linear(params, &format!("{}.self_attn.k_proj", prefix))?;
        let sa_v = load_linear(params, &format!("{}.self_attn.v_proj", prefix))?;
        let sa_out = load_linear(params, &format!("{}.self_attn.out_proj", prefix))?;
        let self_attn =
            SelfAttention::from_weights_with_dim(sa_q, sa_k, sa_v, sa_out, d_model, num_heads);

        let self_attn_ln =
            load_layer_norm(params, &format!("{}.self_attn_layer_norm", prefix), ln_eps)?;

        // Cross-attention (deformable)
        let ea_prefix = format!("{}.encoder_attn", prefix);
        let value_proj = load_linear(params, &format!("{}.value_proj", ea_prefix))?;
        let sampling_offsets = load_linear(params, &format!("{}.sampling_offsets", ea_prefix))?;
        let attn_weights = load_linear(params, &format!("{}.attention_weights", ea_prefix))?;
        let output_proj = load_linear(params, &format!("{}.output_proj", ea_prefix))?;
        let encoder_attn = MultiscaleDeformableAttention::from_weights(
            value_proj,
            sampling_offsets,
            attn_weights,
            output_proj,
            d_model,
            num_heads,
            n_levels,
            n_points,
        );

        let encoder_attn_ln = load_layer_norm(
            params,
            &format!("{}.encoder_attn_layer_norm", prefix),
            ln_eps,
        )?;

        // FFN (fc1/fc2 are directly on the layer, not under mlp. prefix)
        let fc1 = load_linear(params, &format!("{}.fc1", prefix))?;
        let fc2 = load_linear(params, &format!("{}.fc2", prefix))?;
        let final_ln = load_layer_norm(params, &format!("{}.final_layer_norm", prefix), ln_eps)?;

        decoder_layers.push(DecoderLayer::from_components(
            self_attn,
            self_attn_ln,
            encoder_attn,
            encoder_attn_ln,
            fc1,
            fc2,
            final_ln,
            dec_act,
        ));
    }

    // Query position head
    let query_pos_head = load_mlp_head(params, "decoder.query_pos_head", 2)?;

    // Bbox embed (shared with enc_bbox_head via weight tying in checkpoint)
    let bbox_embed = load_mlp_head(params, "enc_bbox_head", 3)?;

    // Class embed (shared with enc_score_head via weight tying in checkpoint)
    let class_embed = load_linear(params, "enc_score_head")?;

    Ok(Decoder::from_components(
        decoder_layers,
        query_pos_head,
        Some(bbox_embed),
        Some(class_embed),
        config.num_queries,
    ))
}

// ============================================================================
// Main Load Function
// ============================================================================

/// Load the full PP-DocLayoutV3 model from a directory.
///
/// The directory must be cloned from `PaddlePaddle/PP-DocLayoutV3_safetensors` on HuggingFace
/// and contain:
/// - `config.json`: Model configuration
/// - `model.safetensors`: Model weights in SafeTensors format
///
/// Note: The regular `PaddlePaddle/PP-DocLayoutV3` repo uses PaddlePaddle format and is not compatible.
///
/// # Arguments
/// * `model_path` - Path to model directory
///
/// # Returns
/// * Fully initialized PPDocLayoutV3Model
pub fn load_model(model_path: &str) -> Result<PPDocLayoutV3Model> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    // Load config
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Err(Error::from_reason(format!(
            "Config file not found: {}",
            config_path.display()
        )));
    }

    let config_data = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;

    let mut config: PPDocLayoutV3Config = serde_json::from_str(&config_data)
        .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

    // Derive num_labels from id2label if not explicitly set (matches HuggingFace behavior)
    config.fixup_after_load();

    // Validate num_feature_levels == 3 (persistence and model inference hardcode 3 levels)
    if config.num_feature_levels != 3 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "PP-DocLayoutV3 only supports num_feature_levels=3, got {}",
                config.num_feature_levels
            ),
        ));
    }
    if config.encoder_in_channels.len() != 3 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "PP-DocLayoutV3 requires encoder_in_channels to have exactly 3 entries, got {}",
                config.encoder_in_channels.len()
            ),
        ));
    }
    if config.decoder_in_channels.len() != 3 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "PP-DocLayoutV3 requires decoder_in_channels to have exactly 3 entries, got {}",
                config.decoder_in_channels.len()
            ),
        ));
    }

    // Load weights
    let raw_params = load_all_safetensors(path, false)?;

    // Strip "model." prefix from all keys
    let mut params: HashMap<String, MxArray> = HashMap::with_capacity(raw_params.len());
    for (key, value) in raw_params {
        let mapped_key = key.strip_prefix("model.").unwrap_or(&key).to_string();
        params.insert(mapped_key, value);
    }

    // Build all components
    let backbone = load_hgnetv2_backbone(
        &params,
        &config.backbone_config,
        "backbone.model.embedder",
        "backbone.model.encoder.stages",
        config.batch_norm_eps,
    )?;

    // Encoder input projections (3 levels: stages 2, 3, 4)
    let mut encoder_input_proj = Vec::with_capacity(3);
    for i in 0..3 {
        let proj = load_input_projection(
            &params,
            &format!("encoder_input_proj.{}", i),
            config.batch_norm_eps,
        )?;
        encoder_input_proj.push(proj);
    }

    let encoder = load_hybrid_encoder(&params, &config)?;

    // Encoder output head
    let enc_output_linear = load_linear(&params, "enc_output.0")?;
    let enc_output_norm = load_layer_norm(&params, "enc_output.1", config.layer_norm_eps)?;

    let enc_score_head = load_linear(&params, "enc_score_head")?;
    let enc_bbox_head = load_mlp_head(&params, "enc_bbox_head", 3)?;

    // Decoder input projections
    let mut decoder_input_proj = Vec::with_capacity(3);
    for i in 0..3 {
        let proj = load_input_projection(
            &params,
            &format!("decoder_input_proj.{}", i),
            config.batch_norm_eps,
        )?;
        decoder_input_proj.push(proj);
    }

    let decoder = load_decoder(&params, &config)?;

    // Decoder order heads (one per decoder layer)
    let mut decoder_order_head = Vec::with_capacity(config.decoder_layers as usize);
    for i in 0..config.decoder_layers {
        let head = load_linear(&params, &format!("decoder_order_head.{}", i))?;
        decoder_order_head.push(head);
    }

    // Global pointer
    let gp_dense = load_linear(&params, "decoder_global_pointer.dense")?;
    let decoder_global_pointer =
        GlobalPointer::from_weights(gp_dense, config.global_pointer_head_size)?;

    // Decoder norm
    let decoder_norm = load_layer_norm(&params, "decoder_norm", config.layer_norm_eps)?;

    // Mask query head
    let mask_query_head = load_mlp_head(&params, "mask_query_head", 3)?;

    // Denoising class embed (optional, not used in inference)
    let denoising_class_embed = params.get("denoising_class_embed.weight").cloned();

    Ok(create_model(
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
        denoising_class_embed,
    ))
}
