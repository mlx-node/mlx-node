/// SFT Training Engine - Rust-native Training Loop
///
/// This module provides a complete Supervised Fine-Tuning (SFT) engine that runs
/// entirely in Rust, eliminating FFI overhead for the core training loop.
///
/// ## Key Features
/// - Simple cross-entropy loss on completion tokens
/// - Gradient accumulation and clipping
/// - Memory management with heavy cleanup
/// - NaN gradient protection
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::{MxArray, heavy_cleanup, synchronize_and_clear_cache};
use crate::models::qwen3::{Qwen3Config, Qwen3Model};
use crate::optimizers::GradientUtils;
use crate::sft::SftLossConfig;
use crate::sft::autograd::{compute_sft_loss_and_gradients, compute_token_accuracy};

/// Configuration for the SFT training engine
#[napi(object)]
#[derive(Clone)]
pub struct SftEngineConfig {
    /// Learning rate (default: 2e-5)
    pub learning_rate: Option<f64>,
    /// Gradient accumulation steps (default: 1)
    pub gradient_accumulation_steps: Option<i32>,
    /// Maximum gradient norm for clipping (default: 1.0)
    pub gradient_clip_norm: Option<f64>,
    /// Maximum gradient value for element-wise clipping (optional)
    pub gradient_clip_value: Option<f64>,
    /// Weight decay (L2 regularization) (default: 0.01)
    pub weight_decay: Option<f64>,
    /// Label smoothing factor (default: 0.0)
    pub label_smoothing: Option<f64>,
    /// Steps between heavy cleanup (default: 25)
    pub heavy_cleanup_interval: Option<i32>,
    /// Maximum allowed NaN gradient occurrences (default: 100)
    pub max_nan_gradients: Option<i64>,
    /// Consecutive NaN gradients that trigger emergency checkpoint (default: 5)
    pub emergency_save_threshold: Option<i32>,
    /// Compute token accuracy (requires extra forward pass) (default: false)
    pub compute_accuracy: Option<bool>,
}

impl Default for SftEngineConfig {
    fn default() -> Self {
        Self {
            learning_rate: Some(2e-5),
            gradient_accumulation_steps: Some(1),
            gradient_clip_norm: Some(1.0),
            gradient_clip_value: None,
            weight_decay: Some(0.01),
            label_smoothing: Some(0.0),
            heavy_cleanup_interval: Some(25),
            max_nan_gradients: Some(100),
            emergency_save_threshold: Some(5),
            compute_accuracy: Some(false),
        }
    }
}

/// Metrics from a single training step
#[napi(object)]
#[derive(Clone)]
pub struct SftStepMetrics {
    /// Current step number
    pub step: i64,
    /// Cross-entropy loss value
    pub loss: f64,
    /// Total tokens processed this step (non-ignored)
    pub total_tokens: i32,
    /// Token-level accuracy (if compute_accuracy enabled)
    pub token_accuracy: Option<f64>,
    /// Whether gradients were applied (vs accumulated)
    pub gradients_applied: bool,
    /// Time for training step (ms)
    pub training_time_ms: f64,
}

/// Metrics from a training epoch
#[napi(object)]
#[derive(Clone)]
pub struct SftEpochMetrics {
    /// Epoch number
    pub epoch: i32,
    /// Average loss for the epoch
    pub avg_loss: f64,
    /// Total steps in the epoch
    pub total_steps: i64,
    /// Total tokens processed
    pub total_tokens: i64,
    /// Time for the epoch (seconds)
    pub epoch_time_secs: f64,
}

/// Result of resume position computation
#[napi(object)]
#[derive(Clone)]
pub struct ResumePosition {
    /// Epoch to start from (0-indexed)
    pub start_epoch: i32,
    /// Batch index within epoch to start from
    pub start_batch_idx: i32,
    /// Whether we're at an epoch boundary
    pub is_epoch_boundary: bool,
}

/// Internal training state
struct EngineState {
    accumulated_gradients: Option<HashMap<String, MxArray>>,
    micro_step: i32,
    step: i64,
    epoch: i32,
    epoch_loss_sum: f64,
    epoch_steps: i64,
    epoch_tokens: i64,
    last_heavy_cleanup_step: i64,
    nan_gradient_count: u64,
    consecutive_nan_count: u32,
    needs_emergency_save: bool,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            accumulated_gradients: None,
            micro_step: 0,
            step: 0,
            epoch: 0,
            epoch_loss_sum: 0.0,
            epoch_steps: 0,
            epoch_tokens: 0,
            last_heavy_cleanup_step: 0,
            nan_gradient_count: 0,
            consecutive_nan_count: 0,
            needs_emergency_save: false,
        }
    }
}

/// SFT Training Engine
#[napi]
pub struct SftTrainingEngine {
    model: Arc<RwLock<Qwen3Model>>,
    model_config: Qwen3Config,
    config: SftEngineConfig,
    state: Arc<RwLock<EngineState>>,
}

#[napi]
impl SftTrainingEngine {
    /// Create a new SFT training engine
    #[napi(constructor)]
    pub fn new(model: &Qwen3Model, config: SftEngineConfig) -> Result<Self> {
        let model_config = model.get_config();

        info!(
            "Creating SFT training engine: {} layers, {} hidden, lr={}",
            model_config.num_layers,
            model_config.hidden_size,
            config.learning_rate.unwrap_or(2e-5)
        );

        Ok(Self {
            model: Arc::new(RwLock::new(model.clone_for_session()?)),
            model_config,
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Run a single training step
    #[napi]
    pub async fn train_step(
        &self,
        input_ids: &MxArray,
        labels: &MxArray,
    ) -> Result<SftStepMetrics> {
        let training_start = std::time::Instant::now();

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let model_config = self.model_config.clone();
        let config = self.config.clone();
        let input_ids = input_ids.clone();
        let labels = labels.clone();

        // Run in spawn_blocking to avoid blocking the async runtime
        let result: std::result::Result<SftStepMetrics, Error> =
            tokio::task::spawn_blocking(move || {
                // Get model parameters
                let params = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.get_parameters()
                };

                // Build loss config
                let loss_config = SftLossConfig {
                    ignore_index: Some(-100),
                    label_smoothing: config.label_smoothing,
                };

                // Compute loss and gradients
                let (loss_value, gradients) = compute_sft_loss_and_gradients(
                    &model_config,
                    &params,
                    &input_ids,
                    &labels,
                    loss_config,
                )?;

                // Check for NaN loss
                if loss_value.is_nan() || loss_value.is_infinite() {
                    let mut state = state_arc.write().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                    })?;
                    state.nan_gradient_count += 1;
                    state.consecutive_nan_count += 1;

                    let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold {
                        state.needs_emergency_save = true;
                        warn!(
                            "Emergency save triggered: {} consecutive NaN losses",
                            state.consecutive_nan_count
                        );
                    }

                    let max_nan = config.max_nan_gradients.unwrap_or(100) as u64;
                    if state.nan_gradient_count >= max_nan {
                        return Err(Error::new(
                            Status::GenericFailure,
                            format!("Training stopped: exceeded {} NaN gradient limit", max_nan),
                        ));
                    }

                    warn!(
                        "NaN loss detected, skipping step (count: {})",
                        state.nan_gradient_count
                    );

                    return Ok(SftStepMetrics {
                        step: state.step,
                        loss: 0.0,
                        total_tokens: 0,
                        token_accuracy: None,
                        gradients_applied: false,
                        training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                    });
                }

                // Reset consecutive NaN count on successful step
                {
                    let mut state = state_arc.write().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                    })?;
                    state.consecutive_nan_count = 0;
                }

                // Count non-ignored tokens
                let total_tokens = count_valid_tokens(&labels)?;

                // Gradient accumulation
                let gradient_accumulation_steps = config.gradient_accumulation_steps.unwrap_or(1);
                let gradients_applied;
                let current_step;

                {
                    let mut state = state_arc.write().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                    })?;

                    // Accumulate gradients
                    if let Some(ref mut acc) = state.accumulated_gradients {
                        for (name, grad) in &gradients {
                            if let Some(acc_grad) = acc.get_mut(name) {
                                *acc_grad = acc_grad.add(grad)?;
                            }
                        }
                    } else {
                        state.accumulated_gradients = Some(gradients.clone());
                    }

                    state.micro_step += 1;

                    // Apply gradients if we've accumulated enough
                    if state.micro_step >= gradient_accumulation_steps {
                        let accumulated = state.accumulated_gradients.take().unwrap();

                        // Average gradients
                        let scale = 1.0 / gradient_accumulation_steps as f64;
                        let averaged: HashMap<String, MxArray> = accumulated
                            .into_iter()
                            .map(|(name, grad)| {
                                let scaled = grad.mul_scalar(scale).unwrap_or(grad);
                                (name, scaled)
                            })
                            .collect();

                        // Clip gradients by global norm
                        let clipped = if let Some(clip_norm) = config.gradient_clip_norm {
                            let grad_refs: HashMap<String, &MxArray> =
                                averaged.iter().map(|(k, v)| (k.clone(), v)).collect();
                            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
                        } else {
                            averaged
                        };

                        // Apply element-wise clipping if configured
                        let final_grads = if let Some(clip_val) = config.gradient_clip_value {
                            clipped
                                .into_iter()
                                .map(|(name, grad)| {
                                    let clipped_grad =
                                        grad.clip(Some(-clip_val), Some(clip_val)).unwrap_or(grad);
                                    (name, clipped_grad)
                                })
                                .collect()
                        } else {
                            clipped
                        };

                        // Apply gradients with weight decay
                        let lr = config.learning_rate.unwrap_or(2e-5);
                        let weight_decay = config.weight_decay.unwrap_or(0.0);

                        // Update model parameters
                        {
                            let mut model = model_arc.write().map_err(|_| {
                                Error::new(
                                    Status::GenericFailure,
                                    "Failed to acquire model write lock",
                                )
                            })?;

                            // Apply weight decay to gradients if configured
                            let grads_with_decay = if weight_decay > 0.0 {
                                let current_params = model.get_parameters();
                                final_grads
                                    .into_iter()
                                    .map(|(name, grad)| {
                                        if let Some(param) = current_params.get(&name) {
                                            // grad_with_decay = grad + weight_decay * param
                                            if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                                && let Ok(new_grad) = grad.add(&decay_term)
                                            {
                                                return (name, new_grad);
                                            }
                                            (name, grad)
                                        } else {
                                            (name, grad)
                                        }
                                    })
                                    .collect::<HashMap<_, _>>()
                            } else {
                                final_grads
                            };

                            let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                                .iter()
                                .map(|(k, v)| (k.clone(), v))
                                .collect();
                            model.apply_gradients(grads_refs, lr)?;
                        }

                        state.step += 1;
                        state.micro_step = 0;
                        gradients_applied = true;

                        // Memory management
                        let cleanup_interval = config.heavy_cleanup_interval.unwrap_or(25) as i64;
                        if state.step - state.last_heavy_cleanup_step >= cleanup_interval {
                            debug!("Heavy cleanup at step {}", state.step);
                            heavy_cleanup();
                            state.last_heavy_cleanup_step = state.step;
                        } else {
                            synchronize_and_clear_cache();
                        }
                    } else {
                        gradients_applied = false;
                        synchronize_and_clear_cache();
                    }

                    // Update epoch metrics
                    state.epoch_loss_sum += loss_value;
                    state.epoch_steps += 1;
                    state.epoch_tokens += total_tokens as i64;
                    current_step = state.step;
                }

                // Compute token accuracy if enabled (requires extra forward pass)
                let token_accuracy = if config.compute_accuracy.unwrap_or(false) {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    let params = model.get_parameters();
                    match compute_token_accuracy(&model_config, &params, &input_ids, &labels) {
                        Ok(acc) => Some(acc),
                        Err(e) => {
                            warn!("Failed to compute accuracy: {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;

                Ok(SftStepMetrics {
                    step: current_step,
                    loss: loss_value,
                    total_tokens,
                    token_accuracy,
                    gradients_applied,
                    training_time_ms,
                })
            })
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {}", e)))?;

        result
    }

    /// Get current step number
    #[napi]
    pub fn get_step(&self) -> Result<i64> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        Ok(state.step)
    }

    /// Get current epoch
    #[napi]
    pub fn get_epoch(&self) -> Result<i32> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        Ok(state.epoch)
    }

    /// Flush any accumulated gradients at epoch end
    ///
    /// When stepsPerEpoch % gradient_accumulation_steps != 0, there may be
    /// leftover gradients from the final micro-batches. This method applies
    /// them with proper averaging, matching TRL behavior.
    #[napi]
    pub fn flush_gradients(&self) -> Result<bool> {
        let model_arc = Arc::clone(&self.model);
        let config = self.config.clone();

        let mut state = self.state.write().map_err(|_| {
            Error::new(Status::GenericFailure, "Failed to acquire state write lock")
        })?;

        // Nothing to flush if no accumulated gradients
        if state.micro_step == 0 || state.accumulated_gradients.is_none() {
            return Ok(false);
        }

        let accumulated = state.accumulated_gradients.take().unwrap();
        let actual_micro_steps = state.micro_step;

        // Average gradients by ACTUAL micro-step count (not configured accumulation steps)
        let scale = 1.0 / actual_micro_steps as f64;
        let averaged: HashMap<String, MxArray> = accumulated
            .into_iter()
            .map(|(name, grad)| {
                let scaled = grad.mul_scalar(scale).unwrap_or(grad);
                (name, scaled)
            })
            .collect();

        // Apply same clipping and weight decay as regular train_step
        let clipped = if let Some(clip_norm) = config.gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> =
                averaged.iter().map(|(k, v)| (k.clone(), v)).collect();
            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
        } else {
            averaged
        };

        let final_grads = if let Some(clip_val) = config.gradient_clip_value {
            clipped
                .into_iter()
                .map(|(name, grad)| {
                    let clipped_grad = grad.clip(Some(-clip_val), Some(clip_val)).unwrap_or(grad);
                    (name, clipped_grad)
                })
                .collect()
        } else {
            clipped
        };

        // Apply gradients
        let lr = config.learning_rate.unwrap_or(2e-5);
        let weight_decay = config.weight_decay.unwrap_or(0.0);

        {
            let mut model = model_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire model write lock")
            })?;

            let grads_with_decay = if weight_decay > 0.0 {
                let current_params = model.get_parameters();
                final_grads
                    .into_iter()
                    .map(|(name, grad)| {
                        if let Some(param) = current_params.get(&name) {
                            if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                && let Ok(new_grad) = grad.add(&decay_term)
                            {
                                return (name, new_grad);
                            }
                            (name, grad)
                        } else {
                            (name, grad)
                        }
                    })
                    .collect::<HashMap<_, _>>()
            } else {
                final_grads
            };

            let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            model.apply_gradients(grads_refs, lr)?;
        }

        state.step += 1;
        state.micro_step = 0;

        info!(
            "Flushed {} micro-batches at epoch end, step now {}",
            actual_micro_steps, state.step
        );

        synchronize_and_clear_cache();
        Ok(true)
    }

    /// Compute the resume position given current state and dataset info
    ///
    /// This centralizes all resume logic in Rust for correctness.
    /// Uses i64 math internally to avoid overflow on long runs.
    #[napi]
    pub fn compute_resume_position(&self, steps_per_epoch: i32) -> Result<ResumePosition> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;

        // Use i64 throughout to avoid overflow on long runs
        let steps_per_epoch = steps_per_epoch as i64;
        let grad_accum = self.config.gradient_accumulation_steps.unwrap_or(1) as i64;

        // With flushGradients(), each epoch has ceil(steps_per_epoch / grad_accum) optimizer steps
        let steps_per_epoch_applied = (steps_per_epoch + grad_accum - 1) / grad_accum; // ceil division

        let current_step = state.step;
        let current_epoch = state.epoch as i64;

        // Compute within-epoch position
        let within_epoch_steps = current_step - current_epoch * steps_per_epoch_applied;

        // Epoch boundary: completed all optimizer steps for current epoch
        let is_epoch_boundary = current_step > 0 && within_epoch_steps >= steps_per_epoch_applied;

        let start_epoch = if is_epoch_boundary {
            current_epoch + 1
        } else {
            current_epoch
        };
        let effective_within = if is_epoch_boundary {
            0
        } else {
            within_epoch_steps
        };
        let start_batch_idx = effective_within * grad_accum;

        // Clamp to i32 for return (batch_idx is bounded by steps_per_epoch which fits in i32)
        Ok(ResumePosition {
            start_epoch: start_epoch as i32,
            start_batch_idx: start_batch_idx as i32,
            is_epoch_boundary,
        })
    }

    /// Check if emergency save is needed
    #[napi]
    pub fn needs_emergency_save(&self) -> Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        Ok(state.needs_emergency_save)
    }

    /// Clear emergency save flag
    #[napi]
    pub fn clear_emergency_save(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        state.needs_emergency_save = false;
        Ok(())
    }

    /// Signal start of a new epoch
    ///
    /// Takes the epoch number directly from TypeScript to ensure synchronization.
    /// The epoch is 0-indexed to match the TypeScript training loop.
    #[napi]
    pub fn start_epoch(&self, epoch: i32) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        state.epoch = epoch;
        state.epoch_loss_sum = 0.0;
        state.epoch_steps = 0;
        state.epoch_tokens = 0;
        info!("Starting epoch {}", state.epoch);
        Ok(())
    }

    /// End current epoch and return metrics
    #[napi]
    pub fn end_epoch(&self, epoch_time_secs: f64) -> Result<SftEpochMetrics> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;

        let avg_loss = if state.epoch_steps > 0 {
            state.epoch_loss_sum / state.epoch_steps as f64
        } else {
            0.0
        };

        info!(
            "Epoch {} complete: avg_loss={:.4}, steps={}, tokens={}",
            state.epoch, avg_loss, state.epoch_steps, state.epoch_tokens
        );

        Ok(SftEpochMetrics {
            epoch: state.epoch,
            avg_loss,
            total_steps: state.epoch_steps,
            total_tokens: state.epoch_tokens,
            epoch_time_secs,
        })
    }

    /// Reset training state (for new training run)
    #[napi]
    pub fn reset(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        *state = EngineState::default();
        info!("Training state reset");
        Ok(())
    }

    /// Restore training state (for resuming from checkpoint)
    #[napi]
    pub fn restore_state(&self, step: i64, epoch: i32) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        state.step = step;
        state.epoch = epoch;
        info!("Restored training state: step={}, epoch={}", step, epoch);
        Ok(())
    }

    /// Get the underlying model for checkpointing
    #[napi]
    pub fn get_model(&self) -> Result<Qwen3Model> {
        let model = self
            .model
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        model.clone_for_session()
    }
}

/// Count tokens that are not ignored (label != -100)
fn count_valid_tokens(labels: &MxArray) -> Result<i32> {
    let shape = labels.shape()?;
    let total: i64 = shape.iter().product();

    // Create mask for non-ignored tokens
    let ignore_val = MxArray::scalar_int(-100)?;
    let valid_mask = labels.not_equal(&ignore_val)?;

    // Sum to count valid tokens
    let count = valid_mask.sum(None, Some(false))?;
    count.eval();

    let count_val = count.item_at_int32(0).unwrap_or(total as i32);
    Ok(count_val)
}
