//! Text Recognition Weight Loading
//!
//! SafeTensors weight loading for the text recognition model.
//! Reuses shared persistence helpers from pp_doclayout_v3.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::backbone::{
    ConvBNActLight, HGBlock, HGBlockLayer, HGNetV2Backbone, HGNetV2Embeddings, HGNetV2Encoder,
    HGNetV2Stage,
};
use crate::models::pp_doclayout_v3::persistence::{
    load_conv_bn_act, load_conv2d, load_frozen_bn, load_layer_norm, load_linear,
};
use crate::utils::safetensors::SafeTensorsFile;

use super::config::TextRecConfig;
use super::ctc_head::CTCHead;
use super::dictionary::CharDictionary;
use super::model::{TextRecModel, create_model};
use super::svtr_neck::{SVTRConvBN, SVTREncoderLayer, SVTRNeck};

// ============================================================================
// Backbone Loading (reuses HGNetV2 with text-rec config)
// ============================================================================

fn load_backbone(
    params: &HashMap<String, MxArray>,
    config: &TextRecConfig,
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
// SVTR Neck Loading (EncoderWithSVTR architecture)
// ============================================================================

/// Load a SVTRConvBN (conv + batchnorm + activation) from parameters.
fn load_svtr_conv_bn(
    params: &HashMap<String, MxArray>,
    conv_prefix: &str,
    bn_prefix: &str,
    activation: &str,
    stride: (i32, i32),
    padding: (i32, i32),
    groups: i32,
    eps: f64,
) -> Result<SVTRConvBN> {
    let conv = load_conv2d(params, conv_prefix, stride, padding, (1, 1), groups, false)?;
    let bn = load_frozen_bn(params, bn_prefix, eps)?;
    Ok(SVTRConvBN::new(conv, bn, activation))
}

fn load_svtr_neck(params: &HashMap<String, MxArray>, config: &TextRecConfig) -> Result<SVTRNeck> {
    let num_layers = config.svtr_num_layers as usize;
    let num_heads = config.svtr_num_heads;
    let eps = config.batch_norm_eps;

    // Conv1: Conv2d(C → C/8, k=[1,3], pad=[0,1])
    let conv1 = load_svtr_conv_bn(
        params,
        "neck.svtr.conv1.conv",
        "neck.svtr.conv1.norm",
        "swish",
        (1, 1),
        (0, 1),
        1,
        eps,
    )?;

    // Conv2: Conv2d(C/8 → hidden, k=1, pad=0)
    let conv2 = load_svtr_conv_bn(
        params,
        "neck.svtr.conv2.conv",
        "neck.svtr.conv2.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;

    // Transformer encoder layers (pre-norm: norm before sublayer)
    // PaddleOCR uses fused QKV projection (self.qkv) and output projection (self.proj)
    let embed_dim = config.svtr_hidden_dim;
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("neck.svtr.layers.{}", i);

        // Fused QKV: Linear(dim, dim * 3)
        let qkv = load_linear(params, &format!("{}.qkv", prefix))?;
        // Output projection: Linear(dim, dim)
        let proj = load_linear(params, &format!("{}.proj", prefix))?;

        // Pre-norm: LayerNorm applied before attention/FFN
        let norm1 = load_layer_norm(params, &format!("{}.norm1", prefix), 1e-5)?;
        let fc1 = load_linear(params, &format!("{}.fc1", prefix))?;
        let fc2 = load_linear(params, &format!("{}.fc2", prefix))?;
        let norm2 = load_layer_norm(params, &format!("{}.norm2", prefix), 1e-5)?;

        layers.push(SVTREncoderLayer::new(
            qkv, proj, num_heads, embed_dim, norm1, fc1, fc2, norm2,
        ));
    }

    // Final LayerNorm after all transformer blocks (eps=1e-6)
    let final_norm = load_layer_norm(params, "neck.svtr.norm", 1e-6)?;

    // Conv3: Conv2d(hidden → C, k=1, pad=0)
    let conv3 = load_svtr_conv_bn(
        params,
        "neck.svtr.conv3.conv",
        "neck.svtr.conv3.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;

    // Conv4: Conv2d(2C → C/8, k=[1,3], pad=[0,1]) — after concat with shortcut
    let conv4 = load_svtr_conv_bn(
        params,
        "neck.svtr.conv4.conv",
        "neck.svtr.conv4.norm",
        "swish",
        (1, 1),
        (0, 1),
        1,
        eps,
    )?;

    // Conv1x1: Conv2d(C/8 → dims, k=1, pad=0) — final projection
    let conv1x1 = load_svtr_conv_bn(
        params,
        "neck.svtr.conv1x1.conv",
        "neck.svtr.conv1x1.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
    )?;

    Ok(SVTRNeck::new(
        conv1, conv2, layers, final_norm, conv3, conv4, conv1x1,
    ))
}

// ============================================================================
// CTC Head Loading
// ============================================================================

fn load_ctc_head(params: &HashMap<String, MxArray>) -> Result<CTCHead> {
    let fc = load_linear(params, "head.fc")?;
    Ok(CTCHead::new(fc))
}

// ============================================================================
// Main Load Function
// ============================================================================

/// Load a TextRecModel from a directory containing model.safetensors and a dictionary file.
pub fn load_model(model_path: &str, dict_path: &str) -> Result<TextRecModel> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    // Load config
    let config_path = path.join("config.json");
    let config: TextRecConfig = if config_path.exists() {
        let config_data = fs::read_to_string(&config_path)
            .map_err(|e| Error::from_reason(format!("Failed to read config: {e}")))?;
        serde_json::from_str(&config_data)
            .map_err(|e| Error::from_reason(format!("Failed to parse config: {e}")))?
    } else {
        TextRecConfig::default()
    };

    // Load dictionary
    let dictionary = CharDictionary::load(dict_path)?;

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
    let neck = load_svtr_neck(&params, &config)?;
    let head = load_ctc_head(&params)?;

    Ok(create_model(&config, backbone, neck, head, dictionary))
}
