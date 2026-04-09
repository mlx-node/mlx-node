//! Training state that lives on the model thread.
//!
//! When training is active, each model's Inner struct holds an `Option<ModelThreadTrainingState>`.
//! This stores optimizer state, gradient accumulation buffers, and cached generation results
//! (as MxArrays) that are reused between the generate and train phases of GRPO.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::optimizers::AdamW;

/// Training state owned by the model thread.
///
/// Created when `InitTraining` command is received, destroyed when training ends.
/// All MxArray state lives here — never crosses the thread boundary.
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
}

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
        }
    }

    /// Clear cached generation results (called after training step consumes them).
    pub fn clear_generation_cache(&mut self) {
        self.cached_prompt_tokens = None;
        self.cached_completion_tokens = None;
        self.cached_completion_logprobs = None;
    }

    /// Serialize AdamW moment tensors + step to a SafeTensors file.
    ///
    /// Must run on the model thread — the MxArrays in optimizer state belong
    /// to the thread that created them.
    ///
    /// Format:
    /// - metadata.step: i64 as string
    /// - metadata.format: "adamw_optimizer_state"
    /// - tensor "{param}.m": first moment
    /// - tensor "{param}.v": second moment
    ///
    /// SGD (no optimizer) or empty state → no-op, returns Ok(()).
    pub(crate) fn save_optimizer_state_sync(&self, path: &str) -> Result<()> {
        let Some(opt) = self.optimizer.as_ref() else {
            return Ok(());
        };
        let step = opt.get_step();
        let keys = opt.get_state_keys();
        if keys.is_empty() {
            return Ok(());
        }
        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        for key in &keys {
            if let Some(m) = opt.get_first_moment(key.clone()) {
                tensors.insert(format!("{}.m", key), m);
            }
            if let Some(v) = opt.get_second_moment(key.clone()) {
                tensors.insert(format!("{}.v", key), v);
            }
        }
        let metadata = serde_json::json!({
            "step": step.to_string(),
            "format": "adamw_optimizer_state",
        });
        crate::utils::safetensors::save_safetensors(path, &tensors, Some(metadata))
    }

    /// Restore AdamW moment tensors + step from a SafeTensors file.
    ///
    /// SGD (no optimizer) → no-op.
    pub(crate) fn load_optimizer_state_sync(&mut self, path: &str) -> Result<()> {
        let Some(opt) = self.optimizer.as_mut() else {
            return Ok(());
        };
        let st_file = crate::utils::safetensors::SafeTensorsFile::load(path)?;
        // Fail loudly if pointed at the wrong file (e.g. model.safetensors)
        // instead of silently ignoring tensors that don't match the .m/.v scheme.
        if let Some(metadata) = &st_file.metadata
            && let Some(fmt) = metadata.get("format").and_then(|v| v.as_str())
            && fmt != "adamw_optimizer_state"
        {
            return Err(Error::from_reason(format!(
                "Expected optimizer state safetensors (format=adamw_optimizer_state), got format={fmt}"
            )));
        }
        if let Some(metadata) = &st_file.metadata
            && let Some(step_str) = metadata.get("step").and_then(|v| v.as_str())
            && let Ok(step) = step_str.parse::<i64>()
        {
            opt.set_step(step);
        }
        let tensors = st_file.load_tensors(path)?;
        for (tensor_key, array) in &tensors {
            if let Some(param_name) = tensor_key.strip_suffix(".m") {
                opt.set_first_moment(param_name.to_string(), array)?;
            } else if let Some(param_name) = tensor_key.strip_suffix(".v") {
                opt.set_second_moment(param_name.to_string(), array)?;
            }
        }
        Ok(())
    }
}
