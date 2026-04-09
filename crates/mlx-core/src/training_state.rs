//! Training state that lives on the model thread.
//!
//! When training is active, each model's Inner struct holds an `Option<ModelThreadTrainingState>`.
//! This stores optimizer state, gradient accumulation buffers, and cached generation results
//! (as MxArrays) that are reused between the generate and train phases of GRPO.

use std::collections::HashMap;

use crate::array::MxArray;
use crate::optimizers::AdamW;

/// Training state owned by the model thread.
///
/// Created when `InitTraining` command is received, destroyed when training ends.
/// All MxArray state lives here — never crosses the thread boundary.
#[allow(dead_code)] // Infrastructure for Phase 2+ training thread migration
pub(crate) struct ModelThreadTrainingState {
    // === Optimizer ===
    pub optimizer: Option<AdamW>,

    // === Gradient accumulation ===
    pub accumulated_gradients: Option<HashMap<String, MxArray>>,
    pub micro_step: i32,
    pub grad_accumulation_steps: i32,

    // === Step tracking ===
    pub step: i64,

    // === NaN tracking ===
    pub nan_gradient_count: u64,
    pub consecutive_nan_count: u32,

    // === Cached generation results (MxArrays reused by TrainStep) ===
    /// Prompt token arrays cached from GenerateForTraining, consumed by TrainStepGRPO.
    pub cached_prompt_tokens: Option<Vec<MxArray>>,
    /// Completion token arrays cached from GenerateForTraining.
    pub cached_completion_tokens: Option<Vec<MxArray>>,
    /// Completion logprob arrays cached from GenerateForTraining.
    pub cached_completion_logprobs: Option<Vec<MxArray>>,

    // === Config (copied from engine config on init) ===
    pub learning_rate: f64,
    pub gradient_clip_norm: Option<f64>,
    pub gradient_clip_value: Option<f64>,
    pub max_nan_gradients: i64,
    pub emergency_save_threshold: i32,
    pub verbose_nan_detection: bool,
    pub gradient_checkpointing: bool,
    pub weight_decay: f64,
}

#[allow(dead_code)] // Infrastructure for Phase 2+ training thread migration
impl ModelThreadTrainingState {
    /// Create a new training state from engine configuration values.
    pub fn new(
        learning_rate: f64,
        grad_accumulation_steps: i32,
        gradient_clip_norm: Option<f64>,
        gradient_clip_value: Option<f64>,
        max_nan_gradients: i64,
        emergency_save_threshold: i32,
        verbose_nan_detection: bool,
        gradient_checkpointing: bool,
        weight_decay: f64,
        optimizer: Option<AdamW>,
    ) -> Self {
        Self {
            optimizer,
            accumulated_gradients: None,
            micro_step: 0,
            grad_accumulation_steps,
            step: 0,
            nan_gradient_count: 0,
            consecutive_nan_count: 0,
            cached_prompt_tokens: None,
            cached_completion_tokens: None,
            cached_completion_logprobs: None,
            learning_rate,
            gradient_clip_norm,
            gradient_clip_value,
            max_nan_gradients,
            emergency_save_threshold,
            verbose_nan_detection,
            gradient_checkpointing,
            weight_decay,
        }
    }

    /// Clear cached generation results (called after training step consumes them).
    pub fn clear_generation_cache(&mut self) {
        self.cached_prompt_tokens = None;
        self.cached_completion_tokens = None;
        self.cached_completion_logprobs = None;
    }
}
