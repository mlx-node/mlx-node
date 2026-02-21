//! PP-DocLayoutV3 Model
//!
//! Document layout analysis model based on RT-DETR architecture with HGNetV2 backbone.
//! Ported from the HuggingFace Transformers implementation.

use crate::array::MxArray;
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;

pub mod backbone;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod heads;
pub mod model;
pub mod persistence;
pub mod postprocessing;
pub mod processing;

/// Apply activation function by name.
pub fn apply_activation(input: &MxArray, activation: &str) -> Result<MxArray> {
    match activation {
        "relu" => Activations::relu(input),
        "silu" | "swish" => Activations::silu(input),
        "gelu" => Activations::gelu(input),
        "hardswish" | "hard_swish" => Activations::hard_swish(input),
        "hardsigmoid" | "hard_sigmoid" => Activations::hard_sigmoid(input),
        "none" | "identity" => Ok(input.clone()),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unsupported activation: {activation}"),
        )),
    }
}
