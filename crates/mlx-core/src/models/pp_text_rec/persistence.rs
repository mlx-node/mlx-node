//! Text Recognition Weight Loading
//!
//! SafeTensors weight loading for the text recognition model.
//! Reuses shared persistence helpers from `crate::vision::cnn`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::persistence::load_all_safetensors;
use crate::vision::cnn::{load_conv_bn_act, load_hgnetv2_backbone, load_layer_norm, load_linear};

use super::config::TextRecConfig;
use super::ctc_head::CTCHead;
use super::dictionary::CharDictionary;
use super::model::{TextRecModel, create_model};
use super::svtr_neck::{SVTREncoderLayer, SVTRNeck};

// ============================================================================
// SVTR Neck Loading (EncoderWithSVTR architecture)
// ============================================================================

fn load_svtr_neck(params: &HashMap<String, MxArray>, config: &TextRecConfig) -> Result<SVTRNeck> {
    let num_layers = config.svtr_num_layers as usize;
    let num_heads = config.svtr_num_heads;
    let eps = config.batch_norm_eps;

    // Conv1: Conv2d(C → C/8, k=[1,3], pad=[0,1])
    let conv1 = load_conv_bn_act(
        params,
        "neck.svtr.conv1.conv",
        "neck.svtr.conv1.norm",
        "swish",
        (1, 1),
        (0, 1),
        1,
        eps,
        None,
    )?;

    // Conv2: Conv2d(C/8 → hidden, k=1, pad=0)
    let conv2 = load_conv_bn_act(
        params,
        "neck.svtr.conv2.conv",
        "neck.svtr.conv2.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
        None,
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
    let conv3 = load_conv_bn_act(
        params,
        "neck.svtr.conv3.conv",
        "neck.svtr.conv3.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
        None,
    )?;

    // Conv4: Conv2d(2C → C/8, k=[1,3], pad=[0,1]) — after concat with shortcut
    let conv4 = load_conv_bn_act(
        params,
        "neck.svtr.conv4.conv",
        "neck.svtr.conv4.norm",
        "swish",
        (1, 1),
        (0, 1),
        1,
        eps,
        None,
    )?;

    // Conv1x1: Conv2d(C/8 → dims, k=1, pad=0) — final projection
    let conv1x1 = load_conv_bn_act(
        params,
        "neck.svtr.conv1x1.conv",
        "neck.svtr.conv1x1.norm",
        "swish",
        (1, 1),
        (0, 0),
        1,
        eps,
        None,
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
    let params = load_all_safetensors(path, false)?;

    // Build components
    let backbone = load_hgnetv2_backbone(
        &params,
        &config.backbone,
        "backbone",
        "backbone.stages",
        config.batch_norm_eps,
    )?;
    let neck = load_svtr_neck(&params, &config)?;
    let head = load_ctc_head(&params)?;

    Ok(create_model(&config, backbone, neck, head, dictionary))
}
