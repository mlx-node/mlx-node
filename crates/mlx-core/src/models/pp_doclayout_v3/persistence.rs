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
use crate::nn::linear::Linear;
use crate::nn::normalization::LayerNorm;
use crate::utils::safetensors::SafeTensorsFile;

use super::backbone::{
    ConvBNAct, ConvBNActLight, FrozenBatchNorm2d, HGBlock, HGBlockLayer, HGNetV2Backbone,
    HGNetV2Embeddings, HGNetV2Encoder, HGNetV2Stage, LearnableAffineBlock, NativeConv2d,
};
use super::config::PPDocLayoutV3Config;
use super::decoder::{Decoder, DecoderLayer, MultiscaleDeformableAttention, SelfAttention};
use super::encoder::{
    CSPRepLayer, ConvLayer, ConvNormLayer, Encoder, EncoderLayer, EncoderMaskOutput, HybridEncoder,
    InputProjection, MaskFeatFPN, MultiheadAttention, RepVggBlock, ScaleHead, ScaleHeadLayer,
};
use super::heads::{GlobalPointer, MLPPredictionHead};
use super::model::{PPDocLayoutV3Model, create_model};

// ============================================================================
// Helper functions
// ============================================================================

/// Get a tensor from the param map, stripping the "model." prefix if present.
pub(crate) fn get_tensor(params: &HashMap<String, MxArray>, key: &str) -> Result<MxArray> {
    params
        .get(key)
        .cloned()
        .ok_or_else(|| Error::from_reason(format!("Missing weight: {}", key)))
}

/// Transpose a Conv2d weight from PyTorch [O, I, kH, kW] to MLX [O, kH, kW, I].
pub(crate) fn transpose_conv_weight(weight: &MxArray) -> Result<MxArray> {
    weight.transpose(Some(&[0, 2, 3, 1]))
}

/// Load a FrozenBatchNorm2d from parameters.
pub(crate) fn load_frozen_bn(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<FrozenBatchNorm2d> {
    let weight = get_tensor(params, &format!("{}.weight", prefix))?;
    let bias = get_tensor(params, &format!("{}.bias", prefix))?;
    let running_mean = get_tensor(params, &format!("{}.running_mean", prefix))?;
    let running_var = get_tensor(params, &format!("{}.running_var", prefix))?;
    Ok(FrozenBatchNorm2d::new(
        &weight,
        &bias,
        &running_mean,
        &running_var,
        eps,
    ))
}

/// Load a NativeConv2d from parameters (transposing weights from OIHW to OHWI).
pub(crate) fn load_conv2d(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    has_bias: bool,
) -> Result<NativeConv2d> {
    let weight_raw = get_tensor(params, &format!("{}.weight", prefix))?;
    let weight = transpose_conv_weight(&weight_raw)?;
    let bias = if has_bias {
        Some(get_tensor(params, &format!("{}.bias", prefix))?)
    } else {
        None
    };
    Ok(NativeConv2d::new(
        &weight,
        bias.as_ref(),
        stride,
        padding,
        dilation,
        groups,
    ))
}

/// Load a Linear layer from parameters.
pub(crate) fn load_linear(params: &HashMap<String, MxArray>, prefix: &str) -> Result<Linear> {
    let weight = get_tensor(params, &format!("{}.weight", prefix))?;
    let bias = params.get(&format!("{}.bias", prefix));
    Linear::from_weights(&weight, bias)
}

/// Load a LayerNorm from parameters.
pub(crate) fn load_layer_norm(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<LayerNorm> {
    let weight = get_tensor(params, &format!("{}.weight", prefix))?;
    let bias = params.get(&format!("{}.bias", prefix));
    LayerNorm::from_weights(&weight, bias, Some(eps))
}

/// Load a ConvBNAct (Conv + BN + Activation) from the backbone.
pub(crate) fn load_conv_bn_act(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    activation: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    groups: i32,
    eps: f64,
    lab_prefix: Option<&str>,
) -> Result<ConvBNAct> {
    let conv = load_conv2d(params, conv_prefix, stride, padding, (1, 1), groups, false)?;
    let norm = load_frozen_bn(params, bn_prefix, eps)?;
    let lab = if let Some(lp) = lab_prefix {
        let scale = get_tensor(params, &format!("{}.scale", lp))?;
        let bias = get_tensor(params, &format!("{}.bias", lp))?;
        Some(LearnableAffineBlock::new(&scale, &bias))
    } else {
        None
    };
    Ok(ConvBNAct::new(conv, norm, activation, lab))
}

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

/// Load a ConvNormLayer from parameters.
pub(crate) fn load_conv_norm_layer(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    activation: Option<&str>,
    stride: (i32, i32),
    padding: (i32, i32),
    groups: i32,
    eps: f64,
) -> Result<ConvNormLayer> {
    let conv = load_conv2d(params, conv_prefix, stride, padding, (1, 1), groups, false)?;
    let norm = load_frozen_bn(params, bn_prefix, eps)?;
    Ok(ConvNormLayer::new(conv, norm, activation))
}

/// Load a ConvLayer (Conv + BN + Activation, always with activation).
pub(crate) fn load_conv_layer(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    activation: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    groups: i32,
    eps: f64,
) -> Result<ConvLayer> {
    let conv = load_conv2d(params, conv_prefix, stride, padding, (1, 1), groups, false)?;
    let norm = load_frozen_bn(params, bn_prefix, eps)?;
    Ok(ConvLayer::new(conv, norm, activation))
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
// Backbone Loading
// ============================================================================

fn load_backbone(
    params: &HashMap<String, MxArray>,
    config: &PPDocLayoutV3Config,
) -> Result<HGNetV2Backbone> {
    let bc = &config.backbone_config;
    let eps = config.batch_norm_eps;
    let act = bc.hidden_act.as_str();
    let use_lab = bc.use_learnable_affine_block;

    // Load stem (embeddings)
    // HF hierarchy: backbone.model.embedder.{stem1,stem2a,stem2b,stem3,stem4}
    // Each stem is an HGNetV2ConvLayer with .convolution, .normalization, .lab
    let stem_prefix = "backbone.model.embedder";

    let lab0 = format!("{}.stem1.lab", stem_prefix);
    let stem1 = load_conv_bn_act(
        params,
        &format!("{}.stem1.convolution", stem_prefix),
        &format!("{}.stem1.normalization", stem_prefix),
        act,
        (2, 2),
        (1, 1),
        1,
        eps,
        if use_lab { Some(lab0.as_str()) } else { None },
    )?;

    let lab1 = format!("{}.stem2a.lab", stem_prefix);
    let stem2a = load_conv_bn_act(
        params,
        &format!("{}.stem2a.convolution", stem_prefix),
        &format!("{}.stem2a.normalization", stem_prefix),
        act,
        (1, 1),
        (0, 0),
        1,
        eps,
        if use_lab { Some(lab1.as_str()) } else { None },
    )?;

    let lab2 = format!("{}.stem2b.lab", stem_prefix);
    let stem2b = load_conv_bn_act(
        params,
        &format!("{}.stem2b.convolution", stem_prefix),
        &format!("{}.stem2b.normalization", stem_prefix),
        act,
        (1, 1),
        (0, 0),
        1,
        eps,
        if use_lab { Some(lab2.as_str()) } else { None },
    )?;

    let lab3 = format!("{}.stem3.lab", stem_prefix);
    let stem3 = load_conv_bn_act(
        params,
        &format!("{}.stem3.convolution", stem_prefix),
        &format!("{}.stem3.normalization", stem_prefix),
        act,
        bc.stem3_stride,
        (1, 1),
        1,
        eps,
        if use_lab { Some(lab3.as_str()) } else { None },
    )?;

    let lab4 = format!("{}.stem4.lab", stem_prefix);
    let stem4 = load_conv_bn_act(
        params,
        &format!("{}.stem4.convolution", stem_prefix),
        &format!("{}.stem4.normalization", stem_prefix),
        act,
        (1, 1),
        (0, 0),
        1,
        eps,
        if use_lab { Some(lab4.as_str()) } else { None },
    )?;

    let embedder = HGNetV2Embeddings::new(stem1, stem2a, stem2b, stem3, stem4);

    // Load encoder stages
    let mut stages = Vec::with_capacity(bc.num_stages());

    for stage_idx in 0..bc.num_stages() {
        let stage_prefix = format!("backbone.model.encoder.stages.{}", stage_idx);
        let do_downsample = bc.stage_downsample[stage_idx];
        let in_channels = bc.stage_in_channels[stage_idx];
        let out_channels = bc.stage_out_channels[stage_idx];
        let is_light = bc.stage_light_block[stage_idx];
        let kernel_size = bc.stage_kernel_size[stage_idx];
        let num_blocks = bc.stage_num_blocks[stage_idx] as usize;
        let num_layers_per_block = bc.stage_numb_of_layers[stage_idx] as usize;
        let mid_channels = bc.stage_mid_channels[stage_idx];
        let stage_stride = bc.stage_strides.get(stage_idx).copied().unwrap_or((2, 2));

        // Downsample conv (depthwise)
        let downsample = if do_downsample {
            let ds = load_conv_bn_act(
                params,
                &format!("{}.downsample.convolution", stage_prefix),
                &format!("{}.downsample.normalization", stage_prefix),
                "none",
                stage_stride,
                (1, 1),
                in_channels,
                eps,
                None,
            )?;
            Some(ds)
        } else {
            None
        };

        // Load HGBlocks
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let block_prefix = format!("{}.blocks.{}", stage_prefix, block_idx);
            let is_first_block = block_idx == 0;

            // Determine channels for this block
            let block_in = if is_first_block {
                in_channels
            } else {
                out_channels
            };

            // Load conv layers
            let mut layers: Vec<HGBlockLayer> = Vec::with_capacity(num_layers_per_block);
            for layer_idx in 0..num_layers_per_block {
                let layer_prefix = format!("{}.layers.{}", block_prefix, layer_idx);
                let layer_in = if layer_idx == 0 {
                    block_in
                } else {
                    mid_channels
                };
                let kp = (kernel_size / 2, kernel_size / 2); // padding = k // 2

                if is_light {
                    // Light block: conv1 (1x1 pointwise, no activation) + conv2 (kxk depthwise)
                    // HF: HGNetV2ConvLayerLight -> self.conv1 = HGNetV2ConvLayer, self.conv2 = HGNetV2ConvLayer
                    // HGNetV2ConvLayer uses .convolution and .normalization
                    let conv1 = load_conv_bn_act(
                        params,
                        &format!("{}.conv1.convolution", layer_prefix),
                        &format!("{}.conv1.normalization", layer_prefix),
                        "none",
                        (1, 1),
                        (0, 0),
                        1,
                        eps,
                        None,
                    )?;
                    let conv2_lab = format!("{}.conv2.lab", layer_prefix);
                    let conv2 = load_conv_bn_act(
                        params,
                        &format!("{}.conv2.convolution", layer_prefix),
                        &format!("{}.conv2.normalization", layer_prefix),
                        act,
                        (1, 1),
                        kp,
                        mid_channels,
                        eps,
                        if use_lab {
                            Some(conv2_lab.as_str())
                        } else {
                            None
                        },
                    )?;
                    layers.push(HGBlockLayer::Light(ConvBNActLight::new(conv1, conv2)));
                } else {
                    // Standard conv
                    // HF: HGNetV2ConvLayer uses .convolution and .normalization
                    let _ = layer_in; // channels handled by the loaded weights
                    let std_lab = format!("{}.lab", layer_prefix);
                    let conv = load_conv_bn_act(
                        params,
                        &format!("{}.convolution", layer_prefix),
                        &format!("{}.normalization", layer_prefix),
                        act,
                        (1, 1),
                        kp,
                        1,
                        eps,
                        if use_lab {
                            Some(std_lab.as_str())
                        } else {
                            None
                        },
                    )?;
                    layers.push(HGBlockLayer::Standard(conv));
                }
            }

            // Aggregation: squeeze + excitation
            // HF: self.aggregation = nn.Sequential(squeeze_conv, excitation_conv)
            // Each is HGNetV2ConvLayer with .convolution, .normalization, .lab
            let agg_prefix = format!("{}.aggregation", block_prefix);
            let agg_lab0 = format!("{}.0.lab", agg_prefix);
            let agg_squeeze = load_conv_bn_act(
                params,
                &format!("{}.0.convolution", agg_prefix),
                &format!("{}.0.normalization", agg_prefix),
                act,
                (1, 1),
                (0, 0),
                1,
                eps,
                if use_lab {
                    Some(agg_lab0.as_str())
                } else {
                    None
                },
            )?;
            let agg_lab1 = format!("{}.1.lab", agg_prefix);
            let agg_excitation = load_conv_bn_act(
                params,
                &format!("{}.1.convolution", agg_prefix),
                &format!("{}.1.normalization", agg_prefix),
                act,
                (1, 1),
                (0, 0),
                1,
                eps,
                if use_lab {
                    Some(agg_lab1.as_str())
                } else {
                    None
                },
            )?;

            let residual = !is_first_block;
            blocks.push(HGBlock::new(layers, agg_squeeze, agg_excitation, residual));
        }

        stages.push(HGNetV2Stage::new(downsample, blocks));
    }

    let encoder = HGNetV2Encoder::new(stages);
    Ok(HGNetV2Backbone::new(embedder, encoder, bc))
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
        let cnl = load_conv_norm_layer(
            params,
            &format!("{}.conv", prefix),
            &format!("{}.norm", prefix),
            Some(act_fn),
            (1, 1),
            (0, 0),
            1,
            eps,
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
        let cnl = load_conv_norm_layer(
            params,
            &format!("{}.conv", prefix),
            &format!("{}.norm", prefix),
            Some(act_fn),
            (2, 2),
            (1, 1),
            1,
            eps,
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
    let encoder_mask_lateral = load_conv_layer(
        params,
        "encoder.encoder_mask_lateral.convolution",
        "encoder.encoder_mask_lateral.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
    )?;

    // Load encoder_mask_output
    // HF: PPDocLayoutV3EncoderMaskOutput.base_conv = PPDocLayoutV3ConvLayer (uses .convolution, .normalization)
    let base_conv = load_conv_layer(
        params,
        "encoder.encoder_mask_output.base_conv.convolution",
        "encoder.encoder_mask_output.base_conv.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
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
    let conv1 = load_conv_norm_layer(
        params,
        &format!("{}.conv1.conv", prefix),
        &format!("{}.conv1.norm", prefix),
        Some(act_fn),
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;
    let conv2 = load_conv_norm_layer(
        params,
        &format!("{}.conv2.conv", prefix),
        &format!("{}.conv2.norm", prefix),
        Some(act_fn),
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;

    // Load bottleneck RepVGG blocks
    let mut bottlenecks = Vec::new();
    let mut idx = 0;
    loop {
        let key = format!("{}.bottlenecks.{}.conv1.conv.weight", prefix, idx);
        if !params.contains_key(&key) {
            break;
        }
        let rep_conv1 = load_conv_norm_layer(
            params,
            &format!("{}.bottlenecks.{}.conv1.conv", prefix, idx),
            &format!("{}.bottlenecks.{}.conv1.norm", prefix, idx),
            None,
            (1, 1),
            (1, 1),
            1,
            eps,
        )?;
        let rep_conv2 = load_conv_norm_layer(
            params,
            &format!("{}.bottlenecks.{}.conv2.conv", prefix, idx),
            &format!("{}.bottlenecks.{}.conv2.norm", prefix, idx),
            None,
            (1, 1),
            (0, 0),
            1,
            eps,
        )?;
        bottlenecks.push(RepVggBlock::new(rep_conv1, rep_conv2, act_fn));
        idx += 1;
    }

    // Optional conv3
    let conv3_key = format!("{}.conv3.conv.weight", prefix);
    let conv3 = if params.contains_key(&conv3_key) {
        Some(load_conv_norm_layer(
            params,
            &format!("{}.conv3.conv", prefix),
            &format!("{}.conv3.norm", prefix),
            Some(act_fn),
            (1, 1),
            (0, 0),
            1,
            eps,
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
            let conv = load_conv_layer(
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
            )?;
            layers.push(ScaleHeadLayer::Conv(conv));
        } else {
            for j in 0..num_scale_layers {
                // HF: layers[j*2] = PPDocLayoutV3ConvLayer, layers[j*2+1] = nn.Upsample
                let conv = load_conv_layer(
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
                )?;
                layers.push(ScaleHeadLayer::Conv(conv));
                layers.push(ScaleHeadLayer::Upsample);
            }
        }

        scale_heads.push(ScaleHead::new(layers));
    }

    // Output conv
    // HF: PPDocLayoutV3ConvLayer uses .convolution and .normalization
    let output_conv = load_conv_layer(
        params,
        "encoder.mask_feature_head.output_conv.convolution",
        "encoder.mask_feature_head.output_conv.normalization",
        act_fn,
        (1, 1),
        (1, 1),
        1,
        eps,
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
    let weights_path = path.join("model.safetensors");
    if !weights_path.exists() {
        return Err(Error::from_reason(format!(
            "Weights file not found: {}",
            weights_path.display()
        )));
    }

    let st_file = SafeTensorsFile::load(&weights_path)?;
    let raw_params = st_file.load_tensors(&weights_path)?;

    // Strip "model." prefix from all keys
    let mut params: HashMap<String, MxArray> = HashMap::with_capacity(raw_params.len());
    for (key, value) in raw_params {
        let mapped_key = key.strip_prefix("model.").unwrap_or(&key).to_string();
        params.insert(mapped_key, value);
    }

    // Build all components
    let backbone = load_backbone(&params, &config)?;

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
        GlobalPointer::from_weights(gp_dense, config.global_pointer_head_size);

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
