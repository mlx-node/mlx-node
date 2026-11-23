//! Shared Tensor & HGNetV2 Backbone Weight Loaders

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::nn::linear::Linear;
use crate::nn::normalization::LayerNorm;

use super::hgnetv2::{
    HGBlock, HGBlockLayer, HGNetV2Backbone, HGNetV2Config, HGNetV2Embeddings, HGNetV2Encoder,
    HGNetV2Stage,
};
use super::layers::{
    ConvBNAct, ConvBNActLight, FrozenBatchNorm2d, LearnableAffineBlock, NativeConv2d,
    NativeConvTranspose2d,
};

/// Get a required tensor from a parameter map.
pub(crate) fn get_tensor(params: &HashMap<String, MxArray>, key: &str) -> Result<MxArray> {
    params
        .get(key)
        .cloned()
        .ok_or_else(|| Error::from_reason(format!("Missing weight: {}", key)))
}

/// Transpose a Conv2d weight from PyTorch `[O, I, kH, kW]` to MLX `[O, kH, kW, I]`.
pub(crate) fn transpose_conv_weight(weight: &MxArray) -> Result<MxArray> {
    weight.transpose(Some(&[0, 2, 3, 1]))
}

/// Load a `FrozenBatchNorm2d` from parameters.
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

/// Load a `NativeConv2d` from parameters (transposing weights from OIHW to OHWI).
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

/// Load a `NativeConvTranspose2d` from parameters (transposing IOHW -> OHWI).
pub(crate) fn load_conv_transpose2d(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    has_bias: bool,
) -> Result<NativeConvTranspose2d> {
    let weight_raw = get_tensor(params, &format!("{}.weight", prefix))?;
    // ConvTranspose2d weights are (C_in, C_out, kH, kW) = IOHW -> MLX OHWI [1, 2, 3, 0].
    let weight = weight_raw.transpose(Some(&[1, 2, 3, 0]))?;
    let bias = if has_bias {
        Some(get_tensor(params, &format!("{}.bias", prefix))?)
    } else {
        None
    };
    Ok(NativeConvTranspose2d::new(
        &weight,
        bias.as_ref(),
        stride,
        padding,
        (1, 1),
        1,
    ))
}

/// Load a `Linear` layer from parameters.
pub(crate) fn load_linear(params: &HashMap<String, MxArray>, prefix: &str) -> Result<Linear> {
    let weight = get_tensor(params, &format!("{}.weight", prefix))?;
    let bias = params.get(&format!("{}.bias", prefix));
    Linear::from_weights(&weight, bias)
}

/// Load a `LayerNorm` from parameters.
pub(crate) fn load_layer_norm(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<LayerNorm> {
    let weight = get_tensor(params, &format!("{}.weight", prefix))?;
    let bias = params.get(&format!("{}.bias", prefix));
    LayerNorm::from_weights(&weight, bias, Some(eps))
}

/// Load a `ConvBNAct` with full dilation and bias options.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_conv_bn_act_with_options(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    activation: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    has_bias: bool,
    eps: f64,
    lab_prefix: Option<&str>,
) -> Result<ConvBNAct> {
    let conv = load_conv2d(
        params,
        conv_prefix,
        stride,
        padding,
        dilation,
        groups,
        has_bias,
    )?;
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

/// Load a standard `ConvBNAct` (dilation=(1,1), bias=false).
#[allow(clippy::too_many_arguments)]
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
    load_conv_bn_act_with_options(
        params,
        conv_prefix,
        bn_prefix,
        activation,
        stride,
        padding,
        (1, 1),
        groups,
        false,
        eps,
        lab_prefix,
    )
}

/// Load an `HGNetV2Backbone` with parameterized stem/stage key prefixes.
///
/// Shared by `pp_doclayout_v3` (`stem_prefix = "backbone.model.embedder"`,
/// `stage_prefix_base = "backbone.model.encoder.stages"`), `pp_text_det`, and
/// `pp_text_rec` (`stem_prefix = "backbone"`, `stage_prefix_base = "backbone.stages"`).
pub(crate) fn load_hgnetv2_backbone(
    params: &HashMap<String, MxArray>,
    bc: &HGNetV2Config,
    stem_prefix: &str,
    stage_prefix_base: &str,
    bn_eps: f64,
) -> Result<HGNetV2Backbone> {
    let act = bc.hidden_act.as_str();
    let use_lab = bc.use_learnable_affine_block;

    let load_stem = |name: &str, stride: (i32, i32), padding: (i32, i32)| -> Result<ConvBNAct> {
        let lab_key = format!("{}.{}.lab", stem_prefix, name);
        load_conv_bn_act(
            params,
            &format!("{}.{}.convolution", stem_prefix, name),
            &format!("{}.{}.normalization", stem_prefix, name),
            act,
            stride,
            padding,
            1,
            bn_eps,
            if use_lab {
                Some(lab_key.as_str())
            } else {
                None
            },
        )
    };

    let stem1 = load_stem("stem1", (2, 2), (1, 1))?;
    let stem2a = load_stem("stem2a", (1, 1), (0, 0))?;
    let stem2b = load_stem("stem2b", (1, 1), (0, 0))?;
    let stem3 = load_stem("stem3", bc.stem3_stride, (1, 1))?;
    let stem4 = load_stem("stem4", (1, 1), (0, 0))?;

    let embedder = HGNetV2Embeddings::new(stem1, stem2a, stem2b, stem3, stem4);

    let mut stages = Vec::with_capacity(bc.num_stages());
    for stage_idx in 0..bc.num_stages() {
        let stage_prefix = format!("{}.{}", stage_prefix_base, stage_idx);
        let do_downsample = bc.stage_downsample[stage_idx];
        let in_channels = bc.stage_in_channels[stage_idx];
        let is_light = bc.stage_light_block[stage_idx];
        let kernel_size = bc.stage_kernel_size[stage_idx];
        let num_blocks = bc.stage_num_blocks[stage_idx] as usize;
        let num_layers_per_block = bc.stage_numb_of_layers[stage_idx] as usize;
        let mid_channels = bc.stage_mid_channels[stage_idx];
        let stage_stride = bc.stage_strides.get(stage_idx).copied().unwrap_or((2, 2));

        let downsample = if do_downsample {
            Some(load_conv_bn_act(
                params,
                &format!("{}.downsample.convolution", stage_prefix),
                &format!("{}.downsample.normalization", stage_prefix),
                "none",
                stage_stride,
                (1, 1),
                in_channels,
                bn_eps,
                None,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let block_prefix = format!("{}.blocks.{}", stage_prefix, block_idx);

            let mut layers: Vec<HGBlockLayer> = Vec::with_capacity(num_layers_per_block);
            for layer_idx in 0..num_layers_per_block {
                let layer_prefix = format!("{}.layers.{}", block_prefix, layer_idx);
                let kp = (kernel_size / 2, kernel_size / 2);

                if is_light {
                    let conv1 = load_conv_bn_act(
                        params,
                        &format!("{}.conv1.convolution", layer_prefix),
                        &format!("{}.conv1.normalization", layer_prefix),
                        "none",
                        (1, 1),
                        (0, 0),
                        1,
                        bn_eps,
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
                        bn_eps,
                        if use_lab {
                            Some(conv2_lab.as_str())
                        } else {
                            None
                        },
                    )?;
                    layers.push(HGBlockLayer::Light(ConvBNActLight::new(conv1, conv2)));
                } else {
                    let std_lab = format!("{}.lab", layer_prefix);
                    let conv = load_conv_bn_act(
                        params,
                        &format!("{}.convolution", layer_prefix),
                        &format!("{}.normalization", layer_prefix),
                        act,
                        (1, 1),
                        kp,
                        1,
                        bn_eps,
                        if use_lab {
                            Some(std_lab.as_str())
                        } else {
                            None
                        },
                    )?;
                    layers.push(HGBlockLayer::Standard(conv));
                }
            }

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
                bn_eps,
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
                bn_eps,
                if use_lab {
                    Some(agg_lab1.as_str())
                } else {
                    None
                },
            )?;

            let residual = block_idx != 0;
            blocks.push(HGBlock::new(layers, agg_squeeze, agg_excitation, residual));
        }

        stages.push(HGNetV2Stage::new(downsample, blocks));
    }

    let encoder = HGNetV2Encoder::new(stages);
    Ok(HGNetV2Backbone::new(embedder, encoder, bc))
}
