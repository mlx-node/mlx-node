//! Text Detection Weight Loading
//!
//! SafeTensors weight loading for the DBNet text detection model.
//! Reuses shared persistence helpers from `crate::vision::cnn`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::persistence::load_all_safetensors;
use crate::vision::cnn::{
    load_conv_transpose2d, load_conv2d, load_frozen_bn, load_hgnetv2_backbone,
};

use super::config::TextDetConfig;
use super::db_head::{Head, LocalModule, PFHeadLocal};
use super::lkpan::{IntraCLBlock, LKPAN};
use super::model::{TextDetModel, create_model};

// ============================================================================
// LKPAN Loading
// ============================================================================

/// Load a single IntraCLBlock from parameters.
///
/// All convolutions in IntraCLBlock have bias (PaddlePaddle default for Conv2D).
/// reduce_factor=2 for LKPAN (channels=64, reduced=32).
fn load_intracl_block(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<IntraCLBlock> {
    // Channel reduction/expansion 1x1 convs (have bias by default in Paddle)
    let conv1x1_reduce = load_conv2d(
        params,
        &format!("{}.conv1x1_reduce_channel", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
    )?;
    let conv1x1_return = load_conv2d(
        params,
        &format!("{}.conv1x1_return_channel", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
    )?;

    // 7x7 level
    let c_layer_7x7 = load_conv2d(
        params,
        &format!("{}.c_layer_7x7", prefix),
        (1, 1),
        (3, 3),
        (1, 1),
        1,
        true,
    )?;
    let v_layer_7x1 = load_conv2d(
        params,
        &format!("{}.v_layer_7x1", prefix),
        (1, 1),
        (3, 0),
        (1, 1),
        1,
        true,
    )?;
    let q_layer_1x7 = load_conv2d(
        params,
        &format!("{}.q_layer_1x7", prefix),
        (1, 1),
        (0, 3),
        (1, 1),
        1,
        true,
    )?;

    // 5x5 level
    let c_layer_5x5 = load_conv2d(
        params,
        &format!("{}.c_layer_5x5", prefix),
        (1, 1),
        (2, 2),
        (1, 1),
        1,
        true,
    )?;
    let v_layer_5x1 = load_conv2d(
        params,
        &format!("{}.v_layer_5x1", prefix),
        (1, 1),
        (2, 0),
        (1, 1),
        1,
        true,
    )?;
    let q_layer_1x5 = load_conv2d(
        params,
        &format!("{}.q_layer_1x5", prefix),
        (1, 1),
        (0, 2),
        (1, 1),
        1,
        true,
    )?;

    // 3x3 level
    let c_layer_3x3 = load_conv2d(
        params,
        &format!("{}.c_layer_3x3", prefix),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        true,
    )?;
    let v_layer_3x1 = load_conv2d(
        params,
        &format!("{}.v_layer_3x1", prefix),
        (1, 1),
        (1, 0),
        (1, 1),
        1,
        true,
    )?;
    let q_layer_1x3 = load_conv2d(
        params,
        &format!("{}.q_layer_1x3", prefix),
        (1, 1),
        (0, 1),
        (1, 1),
        1,
        true,
    )?;

    // BatchNorm2D (PaddleOCR uses nn.BatchNorm2D)
    let bn = load_frozen_bn(params, &format!("{}.bn", prefix), eps)?;

    Ok(IntraCLBlock::new(
        conv1x1_reduce,
        conv1x1_return,
        c_layer_7x7,
        v_layer_7x1,
        q_layer_1x7,
        c_layer_5x5,
        v_layer_5x1,
        q_layer_1x5,
        c_layer_3x3,
        v_layer_3x1,
        q_layer_1x3,
        bn,
    ))
}

fn load_lkpan(params: &HashMap<String, MxArray>, config: &TextDetConfig) -> Result<LKPAN> {
    let lk = config.large_kernel_size;
    let lk_pad = lk / 2;
    let n_levels = config.backbone.stage_out_channels.len();
    let eps = config.batch_norm_eps;

    // ins_conv: 1x1 Conv2D (no bias) per level -- project backbone channels -> 256
    let mut ins_convs = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        ins_convs.push(load_conv2d(
            params,
            &format!("neck.ins_conv.{}", i),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            false,
        )?);
    }

    // inp_conv: 9x9 Conv2D (no bias) per level -- reduce 256 -> 64
    let mut inp_convs = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        inp_convs.push(load_conv2d(
            params,
            &format!("neck.inp_conv.{}", i),
            (1, 1),
            (lk_pad, lk_pad),
            (1, 1),
            1,
            false,
        )?);
    }

    // pan_head_conv: 3x3 stride-2 Conv2D (no bias) -- n_levels-1
    let mut pan_head_convs = Vec::with_capacity(n_levels - 1);
    for i in 0..n_levels - 1 {
        pan_head_convs.push(load_conv2d(
            params,
            &format!("neck.pan_head_conv.{}", i),
            (2, 2),
            (1, 1),
            (1, 1),
            1,
            false,
        )?);
    }

    // pan_lat_conv: 9x9 Conv2D (no bias) per level
    let mut pan_lat_convs = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        pan_lat_convs.push(load_conv2d(
            params,
            &format!("neck.pan_lat_conv.{}", i),
            (1, 1),
            (lk_pad, lk_pad),
            (1, 1),
            1,
            false,
        )?);
    }

    // Try loading IntraCL blocks (optional, check if weights exist)
    let intracl_key = "neck.incl1.conv1x1_reduce_channel.weight";
    if params.contains_key(intracl_key) {
        // Load 4 IntraCL blocks: incl1, incl2, incl3, incl4
        let mut intracl_blocks = Vec::with_capacity(4);
        for i in 1..=4 {
            intracl_blocks.push(load_intracl_block(params, &format!("neck.incl{}", i), eps)?);
        }
        Ok(LKPAN::with_intracl(
            ins_convs,
            inp_convs,
            pan_head_convs,
            pan_lat_convs,
            intracl_blocks,
        ))
    } else {
        Ok(LKPAN::new(
            ins_convs,
            inp_convs,
            pan_head_convs,
            pan_lat_convs,
        ))
    }
}

// ============================================================================
// DBHead Loading
// ============================================================================

/// Load a single Head branch (binarize or thresh).
fn load_head_branch(params: &HashMap<String, MxArray>, prefix: &str, eps: f64) -> Result<Head> {
    // Conv2d(in_ch->inner_ch, k=3, pad=1, bias=False) + BN
    let conv1 = load_conv2d(
        params,
        &format!("{}.conv1", prefix),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        false,
    )?;
    let bn1 = load_frozen_bn(params, &format!("{}.bn1", prefix), eps)?;

    // ConvTranspose2d(inner_ch->inner_ch, k=2, stride=2) + BN
    let deconv1 =
        load_conv_transpose2d(params, &format!("{}.deconv1", prefix), (2, 2), (0, 0), true)?;
    let bn2 = load_frozen_bn(params, &format!("{}.bn2", prefix), eps)?;

    // ConvTranspose2d(inner_ch->1, k=2, stride=2)
    let deconv2 =
        load_conv_transpose2d(params, &format!("{}.deconv2", prefix), (2, 2), (0, 0), true)?;

    Ok(Head::new(conv1, bn1, deconv1, bn2, deconv2))
}

/// Load LocalModule (refinement branch of PFHeadLocal).
///
/// PaddleOCR's LocalModule:
///   self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act="relu")
///   self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)
///
/// ConvBNLayer = Conv2D(bias=False) + BatchNorm(act=None) -> relu
fn load_local_module(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    eps: f64,
) -> Result<LocalModule> {
    // conv1: Conv2d(in_c+1 -> mid_c, k=3, pad=1, bias=False) + BN
    let conv1 = load_conv2d(
        params,
        &format!("{}.conv1", prefix),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        false,
    )?;
    let bn1 = load_frozen_bn(params, &format!("{}.conv1_bn", prefix), eps)?;

    // conv2: Conv2d(mid_c -> 1, k=1, pad=0, bias=True)
    // PaddleOCR: nn.Conv2D(mid_c, 1, 1, 1, 0) — bias_attr defaults to True
    let conv2 = load_conv2d(
        params,
        &format!("{}.conv2", prefix),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
    )?;

    Ok(LocalModule::new(conv1, bn1, conv2))
}

/// Load PFHeadLocal (DB head with local refinement branch).
fn load_pf_head_local(
    params: &HashMap<String, MxArray>,
    config: &TextDetConfig,
) -> Result<PFHeadLocal> {
    let eps = config.batch_norm_eps;

    // Load the binarize (shrink map) head branch
    let binarize = load_head_branch(params, "head.shrink_map", eps)?;

    // Load the cbn_layer (LocalModule)
    let cbn_layer = load_local_module(params, "head.cbn_layer", eps)?;

    Ok(PFHeadLocal::new(binarize, cbn_layer))
}

// ============================================================================
// Main Load Function
// ============================================================================

/// Load a TextDetModel from a directory containing config.json and model.safetensors.
pub fn load_model(model_path: &str) -> Result<TextDetModel> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    // Load config
    let config_path = path.join("config.json");
    let config: TextDetConfig = if config_path.exists() {
        let config_data = fs::read_to_string(&config_path)
            .map_err(|e| Error::from_reason(format!("Failed to read config: {e}")))?;
        serde_json::from_str(&config_data)
            .map_err(|e| Error::from_reason(format!("Failed to parse config: {e}")))?
    } else {
        TextDetConfig::default()
    };

    // Load weights
    let params = load_all_safetensors(path, false)?;

    // Build components
    let backbone = load_hgnetv2_backbone(
        &params,
        &config.backbone,
        "backbone",
        "backbone.stages",
        config.batch_norm_eps,
    )?;
    let neck = load_lkpan(&params, &config)?;
    let head = load_pf_head_local(&params, &config)?;

    Ok(create_model(config, backbone, neck, head))
}
