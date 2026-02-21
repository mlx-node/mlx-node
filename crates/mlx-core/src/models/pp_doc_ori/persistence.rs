//! PP-LCNet_x1_0 Weight Loading
//!
//! SafeTensors weight loading for the document orientation model.
//! Reuses load_conv2d and load_frozen_bn from pp_doclayout_v3.

use std::collections::HashMap;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::persistence::{get_tensor, load_conv2d, load_frozen_bn};
use crate::utils::safetensors::SafeTensorsFile;

use super::config::PPLCNetConfig;
use super::model::{
    ClassificationHead, ConvBNLayer, DepthwiseSeparable, DocOrientationModel, PPLCNetBackbone,
    SEModule, create_model,
};

/// Load a ConvBNLayer (Conv + BN, activation applied externally).
fn load_conv_bn_layer(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    groups: i32,
    eps: f64,
) -> Result<ConvBNLayer> {
    let conv = load_conv2d(
        params,
        &format!("{}.conv", prefix),
        stride,
        padding,
        (1, 1),
        groups,
        false, // no bias, BN handles it
    )?;
    let bn = load_frozen_bn(params, &format!("{}.bn", prefix), eps)?;
    Ok(ConvBNLayer::new(conv, bn))
}

/// Load an SEModule (two 1x1 convs with bias).
fn load_se_module(params: &HashMap<String, MxArray>, prefix: &str) -> Result<SEModule> {
    // SE conv1 and conv2 are standard Conv2D with bias (not Conv+BN)
    let conv1 = load_conv2d(
        params,
        &format!("{}.conv1", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true, // has bias
    )?;
    let conv2 = load_conv2d(
        params,
        &format!("{}.conv2", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true, // has bias
    )?;
    Ok(SEModule::new(conv1, conv2))
}

/// Load a DepthwiseSeparable block.
fn load_depthwise_separable(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    kernel_size: i32,
    in_channels: i32,
    stride: i32,
    use_se: bool,
    eps: f64,
) -> Result<DepthwiseSeparable> {
    let pad = (kernel_size - 1) / 2;

    // Depthwise conv: groups = in_channels
    let dw_conv = load_conv_bn_layer(
        params,
        &format!("{}.dw_conv", prefix),
        (stride, stride),
        (pad, pad),
        in_channels,
        eps,
    )?;

    // Optional SE module
    let se = if use_se {
        Some(load_se_module(params, &format!("{}.se", prefix))?)
    } else {
        None
    };

    // Pointwise conv: 1x1, groups=1
    let pw_conv = load_conv_bn_layer(
        params,
        &format!("{}.pw_conv", prefix),
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;

    Ok(DepthwiseSeparable::new(dw_conv, se, pw_conv))
}

/// Load the full PP-LCNet_x1_0 document orientation model.
pub fn load_model(model_path: &str) -> Result<DocOrientationModel> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    // Load config (optional - we use default PP-LCNet_x1_0 config)
    let config = PPLCNetConfig::default();
    let eps = 1e-5; // BatchNorm epsilon

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

    // Build backbone
    // Stem: conv1 (3x3, stride=2, 3->16 channels)
    let conv1 = load_conv_bn_layer(&params, "backbone.conv1", (2, 2), (1, 1), 1, eps)?;

    // Block groups (blocks2 through blocks6)
    let mut block_groups = Vec::with_capacity(config.blocks.len());

    for (group_name, block_configs) in &config.blocks {
        let mut blocks = Vec::with_capacity(block_configs.len());
        for (i, &(kernel_size, in_channels, _out_channels, stride, use_se)) in
            block_configs.iter().enumerate()
        {
            let prefix = format!("backbone.{}.{}", group_name, i);
            let block = load_depthwise_separable(
                &params,
                &prefix,
                kernel_size,
                in_channels,
                stride,
                use_se,
                eps,
            )?;
            blocks.push(block);
        }
        block_groups.push(blocks);
    }

    // Optional last_conv (1x1 expansion, e.g. 512 -> 1280, no BN)
    let last_conv = if params.contains_key("backbone.last_conv.conv.weight") {
        Some(load_conv2d(
            &params,
            "backbone.last_conv.conv",
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            false, // no bias
        )?)
    } else {
        None
    };

    let backbone = PPLCNetBackbone::new(conv1, block_groups, last_conv);

    // Classification head: FC layer
    let head_weight = get_tensor(&params, "head.weight")?;
    let head_bias = get_tensor(&params, "head.bias")?;
    let head = ClassificationHead::new(&head_weight, &head_bias);

    Ok(create_model(backbone, head))
}
