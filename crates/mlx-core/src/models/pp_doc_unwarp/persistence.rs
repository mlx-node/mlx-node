//! UVDoc Weight Loading
//!
//! SafeTensors weight loading for the UVDoc model.
//! Weight keys follow the PyTorch UVDoc naming convention.

use std::collections::HashMap;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::persistence::{get_tensor, load_conv2d, load_frozen_bn};
use crate::utils::safetensors::SafeTensorsFile;

use super::config::UVDocConfig;
use super::model::{
    Bridge, ConvBNRelu, DocUnwarpModel, OutputHead, ResLayer, ResidualBlock, ResnetEncoder,
    UVDocNet, create_model,
};

const BN_EPS: f64 = 1e-5;

/// Load a ConvBNRelu block.
fn load_conv_bn_relu(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    has_bias: bool,
) -> Result<ConvBNRelu> {
    let conv = load_conv2d(
        params,
        conv_prefix,
        stride,
        padding,
        dilation,
        groups,
        has_bias,
    )?;
    let bn = load_frozen_bn(params, bn_prefix, BN_EPS)?;
    Ok(ConvBNRelu::new(conv, bn))
}

/// Load a residual block.
/// The UVDoc ResidualBlockWithDilation has conv1, bn1, conv2, bn2, and optional downsample.
fn load_residual_block(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    kernel_size: i32,
    stride: i32,
    is_top: bool,
) -> Result<ResidualBlock> {
    let pad = kernel_size / 2;

    // Determine dilation
    let (dil, actual_pad) = if stride != 1 || is_top {
        // Standard conv with no dilation
        (1, pad)
    } else {
        // Dilated conv with dilation=3
        (3, 3 * pad)
    };

    let actual_stride = if stride != 1 || is_top { stride } else { 1 };

    let conv1 = load_conv2d(
        params,
        &format!("{}.conv1.0", prefix),
        (actual_stride, actual_stride),
        (actual_pad, actual_pad),
        (dil, dil),
        1,
        true,
    )?;
    let bn1 = load_frozen_bn(params, &format!("{}.bn1", prefix), BN_EPS)?;

    let conv2 = load_conv2d(
        params,
        &format!("{}.conv2.0", prefix),
        (1, 1),
        (actual_pad, actual_pad),
        (dil, dil),
        1,
        true,
    )?;
    let bn2 = load_frozen_bn(params, &format!("{}.bn2", prefix), BN_EPS)?;

    // Optional downsample
    let ds_key = format!("{}.downsample.0.weight", prefix);
    let downsample = if params.contains_key(&ds_key) {
        let ds_conv = load_conv2d(
            params,
            &format!("{}.downsample.0", prefix),
            (actual_stride, actual_stride),
            (pad, pad),
            (1, 1),
            1,
            true,
        )?;
        let ds_bn = load_frozen_bn(params, &format!("{}.downsample.1", prefix), BN_EPS)?;
        Some((ds_conv, ds_bn))
    } else {
        None
    };

    Ok(ResidualBlock::new(conv1, bn1, conv2, bn2, downsample))
}

/// Load a bridge branch (sequence of dilated ConvBNRelu blocks).
fn load_bridge_branch(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    dilations: &[i32],
) -> Result<Vec<ConvBNRelu>> {
    let mut layers = Vec::with_capacity(dilations.len());
    for (i, &dil) in dilations.iter().enumerate() {
        let pad = dil; // padding = dilation for kernel_size=3
        let cbr = load_conv_bn_relu(
            params,
            &format!("{}.{}.0", prefix, i),
            &format!("{}.{}.1", prefix, i),
            (1, 1),
            (pad, pad),
            (dil, dil),
            1,
            false,
        )?;
        layers.push(cbr);
    }
    Ok(layers)
}

/// Load the OutputHead (Conv + BN + PReLU + Conv).
///
/// Output head uses reflect padding (applied manually in forward pass),
/// so convolutions are loaded with padding=0.
fn load_output_head(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    kernel_size: i32,
) -> Result<OutputHead> {
    let pad = kernel_size / 2;

    // conv1 (no bias, followed by BN, padding=0 since we use reflect padding)
    let conv1 = load_conv2d(
        params,
        &format!("{}.0", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        false,
    )?;
    let bn = load_frozen_bn(params, &format!("{}.1", prefix), BN_EPS)?;

    // PReLU weight
    let prelu_weight = get_tensor(params, &format!("{}.2.weight", prefix))?;

    // conv2 (has bias, no BN after, padding=0 since we use reflect padding)
    let conv2 = load_conv2d(
        params,
        &format!("{}.3", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
    )?;

    Ok(OutputHead::new(conv1, bn, prelu_weight, conv2, pad as i64))
}

/// Load the full UVDoc model.
pub fn load_model(model_path: &str) -> Result<DocUnwarpModel> {
    let path = Path::new(model_path);
    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    let config = UVDocConfig::default();
    let ks = config.kernel_size;
    let pad = ks / 2;

    // Load weights
    let weights_path = path.join("model.safetensors");
    if !weights_path.exists() {
        return Err(Error::from_reason(format!(
            "Weights file not found: {}",
            weights_path.display()
        )));
    }

    let st_file = SafeTensorsFile::load(&weights_path)?;
    let params = st_file.load_tensors(&weights_path)?;

    // ---- Head: 2x (Conv5x5/s2 + BN + ReLU) ----
    let head_0 = load_conv_bn_relu(
        &params,
        "resnet_head.0",
        "resnet_head.1",
        (2, 2),
        (pad, pad),
        (1, 1),
        1,
        false,
    )?;
    let head_1 = load_conv_bn_relu(
        &params,
        "resnet_head.3",
        "resnet_head.4",
        (2, 2),
        (pad, pad),
        (1, 1),
        1,
        false,
    )?;
    let head = vec![head_0, head_1];

    // ---- Encoder: ResnetStraight with 3 layers ----
    let mut encoder_layers = Vec::with_capacity(3);
    for layer_idx in 0..3usize {
        let layer_prefix = format!("resnet_down.layer{}", layer_idx + 1);
        let block_count = config.block_nums[layer_idx];
        let stride = config.stride[layer_idx];

        let mut blocks = Vec::with_capacity(block_count);
        for block_idx in 0..block_count {
            let block_prefix = format!("{}.{}", layer_prefix, block_idx);
            let is_top = block_idx == 0;
            let block_stride = if block_idx == 0 { stride } else { 1 };
            let block = load_residual_block(&params, &block_prefix, ks, block_stride, is_top)?;
            blocks.push(block);
        }
        encoder_layers.push(ResLayer::new(blocks));
    }
    let encoder = ResnetEncoder::new(encoder_layers);

    // ---- Bridge ----
    let bridge_1 = load_bridge_branch(&params, "bridge_1.0", &[1])?;
    let bridge_2 = load_bridge_branch(&params, "bridge_2.0", &[2])?;
    let bridge_3 = load_bridge_branch(&params, "bridge_3.0", &[5])?;
    let bridge_4 = load_bridge_branch(&params, "bridge_4", &[8, 3, 2])?;
    let bridge_5 = load_bridge_branch(&params, "bridge_5", &[12, 7, 4])?;
    let bridge_6 = load_bridge_branch(&params, "bridge_6", &[18, 12, 6])?;

    // bridge_concat: 1x1 Conv + BN + ReLU
    let bridge_concat_conv = load_conv_bn_relu(
        &params,
        "bridge_concat.0",
        "bridge_concat.1",
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        false,
    )?;

    let bridge = Bridge::new(
        bridge_1,
        bridge_2,
        bridge_3,
        bridge_4,
        bridge_5,
        bridge_6,
        bridge_concat_conv,
    );

    // ---- Output head: 2D displacement field ----
    let out_2d = load_output_head(&params, "out_point_positions2D", ks)?;

    let net = UVDocNet::new(head, encoder, bridge, out_2d);
    Ok(create_model(net, config))
}
