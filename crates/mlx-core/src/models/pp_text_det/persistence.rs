//! Text Detection Weight Loading
//!
//! SafeTensors weight loading for the DBNet text detection model.
//! Reuses shared persistence helpers from pp_doclayout_v3.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::backbone::{
    ConvBNActLight, HGBlock, HGBlockLayer, HGNetV2Backbone, HGNetV2Embeddings, HGNetV2Encoder,
    HGNetV2Stage, NativeConvTranspose2d,
};
use crate::models::pp_doclayout_v3::persistence::{
    get_tensor, load_conv_bn_act, load_conv2d, load_frozen_bn,
};
use crate::utils::safetensors::SafeTensorsFile;

use super::config::TextDetConfig;
use super::db_head::{Head, LocalModule, PFHeadLocal};
use super::lkpan::{IntraCLBlock, LKPAN};
use super::model::{TextDetModel, create_model};

// ============================================================================
// Backbone Loading (reuses HGNetV2 with text-det config)
// ============================================================================

fn load_backbone(
    params: &HashMap<String, MxArray>,
    config: &TextDetConfig,
) -> Result<HGNetV2Backbone> {
    let bc = &config.backbone;
    let eps = config.batch_norm_eps;
    let act = bc.hidden_act.as_str();
    let use_lab = bc.use_learnable_affine_block;

    // Load stem
    let stem_prefix = "backbone";

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
        let stage_prefix = format!("{}.stages.{}", stem_prefix, stage_idx);
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
                eps,
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

            // Aggregation
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

            let residual = block_idx != 0;
            blocks.push(HGBlock::new(layers, agg_squeeze, agg_excitation, residual));
        }

        stages.push(HGNetV2Stage::new(downsample, blocks));
    }

    let encoder = HGNetV2Encoder::new(stages);
    Ok(HGNetV2Backbone::new(embedder, encoder, bc))
}

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

fn load_conv_transpose2d(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    has_bias: bool,
) -> Result<NativeConvTranspose2d> {
    let weight_raw = get_tensor(params, &format!("{}.weight", prefix))?;
    // ConvTranspose2d weights in Paddle are (C_in, C_out, kH, kW) = IOHW,
    // whereas regular Conv2d is (C_out, C_in, kH, kW) = OIHW.
    // MLX expects (C_out, kH, kW, C_in) = OHWI, so we need [1, 2, 3, 0].
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
    let weights_path = path.join("model.safetensors");
    if !weights_path.exists() {
        return Err(Error::from_reason(format!(
            "Weights file not found: {}",
            weights_path.display()
        )));
    }

    let st_file = SafeTensorsFile::load(&weights_path)?;
    let params = st_file.load_tensors(&weights_path)?;

    // Build components
    let backbone = load_backbone(&params, &config)?;
    let neck = load_lkpan(&params, &config)?;
    let head = load_pf_head_local(&params, &config)?;

    Ok(create_model(config, backbone, neck, head))
}
