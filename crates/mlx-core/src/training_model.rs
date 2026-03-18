/// Model-Agnostic Training Support
///
/// Provides a `TrainableModel` trait and `TrainableModelEnum` that allow the GRPO and SFT
/// training engines to work with any supported model family (Qwen3, Qwen3.5 Dense, Qwen3.5 MoE).
///
/// ## Architecture
///
/// NAPI-RS cannot use trait objects or generics in `#[napi]` structs. We solve this with:
/// 1. `TrainableModel` trait (Rust-only) defining the common training interface
/// 2. `TrainableModelEnum` wrapping all model types, stored in engines
/// 3. `ModelType` enum carrying config for functional forward passes in autograd
use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::qwen3::{
    BatchGenerationResult, GenerationConfig, GenerationResult, Qwen3Config, Qwen3Model,
};
use crate::models::qwen3_5::Qwen3_5Config;
use crate::models::qwen3_5::model::Qwen3_5Model;
use crate::models::qwen3_5_moe::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use crate::tokenizer::{ChatMessage, ToolDefinition};

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

/// Dispatch a method call across all TrainableModelEnum variants.
macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match $self {
            TrainableModelEnum::Qwen3(m) => m.$method($($arg),*),
            TrainableModelEnum::Qwen35Dense(m) => m.$method($($arg),*),
            TrainableModelEnum::Qwen35Moe(m) => m.$method($($arg),*),
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

/// Common training interface for all model families.
///
/// This trait is Rust-internal only (not exposed via NAPI). Methods that engines
/// call through the trait during training steps. Methods like `clone_for_training`,
/// `calculate_memory_size`, and `save_model_sync` exist on concrete model types
/// but are NOT in this trait — engines call them on the concrete types before
/// wrapping in `TrainableModelEnum`.
pub(crate) trait TrainableModel: Send + Sync {
    /// Extract all trainable parameters as a name→array map.
    fn get_parameters(&self) -> Result<HashMap<String, MxArray>>;

    /// Apply gradients using SGD: param = param - lr * grad.
    fn apply_gradients_with_params(
        &mut self,
        grads: HashMap<String, &MxArray>,
        lr: f64,
        params: &HashMap<String, MxArray>,
    ) -> Result<()>;

    /// Tokenize messages using the model's chat template.
    fn apply_chat_template_sync(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
        tools: Option<&[ToolDefinition]>,
        enable_thinking: Option<bool>,
    ) -> Result<Vec<u32>>;

    /// Generate a single completion with logprob tracking (for GRPO).
    fn generate_for_training_sync(
        &self,
        input_ids: &MxArray,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult>;

    /// Generate a batch of completions (for GRPO).
    fn generate_batch_for_training_sync(
        &self,
        prompt_arrays: &[MxArray],
        group_size: usize,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult>;

    /// Decode token IDs to text.
    fn decode_tokens_sync(&self, tokens: &MxArray) -> Result<String>;
}

/// Enum wrapping all trainable model types.
///
/// Stored inside training engines to allow model-agnostic training.
pub(crate) enum TrainableModelEnum {
    Qwen3(Qwen3Model),
    Qwen35Dense(Qwen3_5Model),
    Qwen35Moe(Qwen3_5MoeModel),
}

impl TrainableModel for TrainableModelEnum {
    fn get_parameters(&self) -> Result<HashMap<String, MxArray>> {
        match self {
            TrainableModelEnum::Qwen3(m) => m.get_parameters(),
            TrainableModelEnum::Qwen35Dense(m) => m.get_parameters_for_training(),
            TrainableModelEnum::Qwen35Moe(m) => m.get_parameters_for_training(),
        }
    }

    fn apply_gradients_with_params(
        &mut self,
        grads: HashMap<String, &MxArray>,
        lr: f64,
        params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        dispatch!(self, apply_gradients_with_params(grads, lr, params))
    }

    fn apply_chat_template_sync(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
        tools: Option<&[ToolDefinition]>,
        enable_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        dispatch!(
            self,
            apply_chat_template_sync(messages, add_generation_prompt, tools, enable_thinking)
        )
    }

    fn generate_for_training_sync(
        &self,
        input_ids: &MxArray,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        dispatch!(self, generate_for_training_sync(input_ids, config))
    }

    fn generate_batch_for_training_sync(
        &self,
        prompt_arrays: &[MxArray],
        group_size: usize,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult> {
        dispatch!(
            self,
            generate_batch_for_training_sync(prompt_arrays, group_size, config)
        )
    }

    fn decode_tokens_sync(&self, tokens: &MxArray) -> Result<String> {
        dispatch!(self, decode_tokens_sync(tokens))
    }
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
