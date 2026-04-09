/// Model-Agnostic Training Support
///
/// Provides shared types for training engines that route through model threads:
/// - `ModelType` enum carrying config for functional forward passes in autograd
/// - `TrainingDispatch` for sending training commands to model threads
/// - `GenerationPlainData` / `TrainStepPlainMetrics` for cross-thread data
/// - `compute_sgd_updates` shared SGD helper used by all model implementations
use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::qwen3::Qwen3Config;
use crate::models::qwen3_5::Qwen3_5Config;
use crate::models::qwen3_5_moe::Qwen3_5MoeConfig;

/// Dispatch a field access across all ModelType config variants.
macro_rules! config_field {
    ($self:expr, $field:ident) => {
        match $self {
            ModelType::Qwen3(c) => c.$field,
            ModelType::Qwen35Dense(c) => c.$field,
            ModelType::Qwen35Moe(c) => c.$field,
        }
    };
}

/// Model type enum carrying the config needed for functional forward passes.
///
/// This is passed to autograd functions so they can dispatch to the correct
/// functional forward pass implementation.
#[derive(Clone)]
pub(crate) enum ModelType {
    Qwen3(Qwen3Config),
    Qwen35Dense(Qwen3_5Config),
    Qwen35Moe(Qwen3_5MoeConfig),
}

impl ModelType {
    pub fn pad_token_id(&self) -> i32 {
        config_field!(self, pad_token_id)
    }
    pub fn vocab_size(&self) -> i32 {
        config_field!(self, vocab_size)
    }
    pub fn hidden_size(&self) -> i32 {
        config_field!(self, hidden_size)
    }
    pub fn eos_token_id(&self) -> i32 {
        config_field!(self, eos_token_id)
    }
    pub fn tie_word_embeddings(&self) -> bool {
        config_field!(self, tie_word_embeddings)
    }
    pub fn num_layers(&self) -> i32 {
        config_field!(self, num_layers)
    }
}

/// Plain generation results that cross the thread boundary.
/// No MxArrays — only plain Rust types (Vec<u32>, Vec<f32>, String, etc.).
/// The model thread caches the MxArray versions internally for the subsequent training step.
pub(crate) struct GenerationPlainData {
    pub completion_texts: Vec<String>,
    pub prompt_texts: Vec<String>,
    pub completion_tokens: Vec<Vec<i32>>,
    pub completion_logprobs: Vec<Vec<f32>>,
    pub token_counts: Vec<u32>,
    pub finish_reasons: Vec<String>,
}

/// Plain training metrics that cross the thread boundary.
/// No MxArrays — only plain numeric types.
pub(crate) struct TrainStepPlainMetrics {
    pub loss: f64,
    pub gradients_applied: bool,
    pub mean_advantage: f64,
    pub std_advantage: f64,
    pub nan_gradient_count: u64,
    pub peak_memory_mb: f64,
    pub active_memory_mb: f64,
    pub total_tokens: i32,
    pub step: i64,
}

/// Dispatch handle for sending training commands to the appropriate model thread.
/// Training engines hold this to route commands to the correct model's dedicated thread.
pub(crate) enum TrainingDispatch {
    Qwen3(tokio::sync::mpsc::UnboundedSender<crate::models::qwen3::Qwen3Cmd>),
    Qwen35Dense(tokio::sync::mpsc::UnboundedSender<crate::models::qwen3_5::model::Qwen35Cmd>),
    Qwen35Moe(tokio::sync::mpsc::UnboundedSender<crate::models::qwen3_5_moe::model::Qwen35MoeCmd>),
}

/// Compute SGD parameter updates: param = param - lr * grad.
///
/// Shared helper used by all model implementations to avoid duplicating the
/// update logic. Returns a map of parameter name → updated array with all
/// values already eval'd.
pub(crate) fn compute_sgd_updates(
    gradients: &HashMap<String, &MxArray>,
    learning_rate: f64,
    current_params: &HashMap<String, MxArray>,
) -> Result<HashMap<String, MxArray>> {
    let lr_scalar_f32 = MxArray::full(&[], Either::A(learning_rate), None)?;
    let mut updated_params: HashMap<String, MxArray> = HashMap::new();

    for (name, grad) in gradients.iter() {
        let param = current_params.get(name.as_str()).ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                format!("Parameter '{}' not found", name),
            )
        })?;
        let param_dtype = param.dtype()?;
        let lr_scalar = lr_scalar_f32.astype(param_dtype)?;
        let scaled_grad = lr_scalar.mul(grad)?;
        let updated_param = param.sub(&scaled_grad)?;
        let updated_param = if updated_param.dtype()? != param_dtype {
            updated_param.astype(param_dtype)?
        } else {
            updated_param
        };
        updated_params.insert(name.clone(), updated_param);
    }

    // Batch eval all updated parameters
    for param in updated_params.values() {
        param.eval();
    }

    Ok(updated_params)
}
