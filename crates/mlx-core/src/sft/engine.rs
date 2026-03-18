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
use crate::models::qwen3_5::model::Qwen3_5Model;
use crate::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use crate::optimizers::GradientUtils;
use crate::sft::SftLossConfig;
use crate::sft::autograd::{compute_sft_loss_and_gradients, compute_token_accuracy};
use crate::training_model::{ModelType, TrainableModel, TrainableModelEnum};

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
    /// Enable detailed NaN/Inf detection with per-element counts (default: false)
    /// When false (default), uses GPU-native has_nan_or_inf() which only transfers a single
    /// boolean to CPU. When true, transfers the entire gradient tensor to CPU for detailed
    /// per-element analysis - useful for debugging but has significant performance overhead.
    pub verbose_nan_detection: Option<bool>,
    /// Enable gradient checkpointing to reduce memory (default: true)
    /// Trades ~30% more compute for O(1) layer memory instead of O(num_layers).
    pub gradient_checkpointing: Option<bool>,
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
            verbose_nan_detection: Some(false),
            gradient_checkpointing: Some(true),
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
    model: Arc<RwLock<TrainableModelEnum>>,
    model_type: ModelType,
    config: SftEngineConfig,
    state: Arc<RwLock<EngineState>>,
}

#[napi]
impl SftTrainingEngine {
    /// Create a new SFT training engine from a Qwen3 model
    #[napi(constructor)]
    pub fn new(model: &Qwen3Model, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen3(model.get_config());

        // Let MLX manage memory dynamically - no explicit cache limits
        // Previously set cache_limit which conflicted with other limit calls
        // causing 93GB+ startup memory. MLX handles memory better without limits.

        info!(
            "Creating SFT training engine: {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        Ok(Self {
            // clone_for_session() is now O(1) - just clones Arc pointers.
            // The model uses RwLock for interior mutability, so apply_gradients()
            // can acquire write locks without needing unique Arc ownership.
            // This eliminates the previous ~4GB memory overhead from deep cloning.
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen3(
                model.clone_for_session()?,
            ))),
            model_type,
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new SFT training engine from a Qwen3.5 dense model
    #[napi(factory)]
    pub fn from_qwen35(model: &Qwen3_5Model, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Dense(model.get_config());

        info!(
            "Creating SFT training engine (Qwen3.5 Dense): {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        Ok(Self {
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen35Dense(
                model.clone_for_training()?,
            ))),
            model_type,
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new SFT training engine from a Qwen3.5 MoE model
    #[napi(factory)]
    pub fn from_qwen35_moe(model: &Qwen3_5MoeModel, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Moe(model.get_config());

        info!(
            "Creating SFT training engine (Qwen3.5 MoE): {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        Ok(Self {
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen35Moe(
                model.clone_for_training()?,
            ))),
            model_type,
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Estimate memory required for training in GB (comprehensive estimate)
    ///
    /// Components:
    /// - Model weights: num_params * 2 bytes (bf16)
    /// - Gradients: same as weights
    /// - Forward pass intermediates: ~16x hidden_size per layer per token
    /// - Attention weights: batch * heads * seq² * 4 bytes per layer
    ///
    /// Note: This comprehensive estimate was causing MLX to pre-allocate 100GB+.
    /// Use `estimate_model_weights_only` for conservative cache limits instead.
    #[allow(dead_code)]
    fn _estimate_training_memory(config: &Qwen3Config) -> f64 {
        // Estimate number of parameters
        let hidden = config.hidden_size as f64;
        let vocab = config.vocab_size as f64;
        let layers = config.num_layers as f64;
        let intermediate = config.intermediate_size as f64;

        // Per-layer params: attention (4 * hidden²) + MLP (3 * hidden * intermediate) + norms
        let per_layer_params = 4.0 * hidden * hidden + 3.0 * hidden * intermediate + 2.0 * hidden;
        let embedding_params = vocab * hidden;
        let total_params = per_layer_params * layers + embedding_params * 2.0; // embed + lm_head

        // Weights + gradients (bf16 = 2 bytes each)
        let weights_gb = total_params * 2.0 / 1e9;
        let gradients_gb = weights_gb;

        // Forward pass intermediates (rough estimate: 16 bytes per hidden per layer per token)
        // Assuming batch_size=4, seq_len=2048
        let batch = 4.0;
        let seq = 2048.0;
        let intermediates_gb = layers * batch * seq * hidden * 16.0 / 1e9;

        // Attention weights: batch * heads * seq² * sizeof(float) per layer
        let heads = config.num_heads as f64;
        let attention_gb = layers * batch * heads * seq * seq * 4.0 / 1e9;

        let total_gb = weights_gb + gradients_gb + intermediates_gb + attention_gb;

        debug!(
            "Memory estimate: weights={:.1}GB, grads={:.1}GB, intermediates={:.1}GB, attention={:.1}GB, total={:.1}GB",
            weights_gb, gradients_gb, intermediates_gb, attention_gb, total_gb
        );

        total_gb
    }

    /// Estimate model weights only (for conservative cache limit)
    ///
    /// This provides a minimal memory estimate based only on model parameters,
    /// avoiding aggressive pre-allocation that can cause 100GB+ memory usage.
    /// Kept for potential future diagnostics.
    #[allow(dead_code)]
    fn estimate_model_weights_only(config: &Qwen3Config) -> f64 {
        let hidden_size = config.hidden_size as f64;
        let intermediate_size = config.intermediate_size as f64;
        let num_layers = config.num_layers as f64;
        let vocab_size = config.vocab_size as f64;

        // Parameters per layer: attention (4 projections) + MLP (3 projections) + norms
        let attention_params = 4.0 * hidden_size * hidden_size;
        let mlp_params = 3.0 * hidden_size * intermediate_size;
        let norm_params = 2.0 * hidden_size;
        let layer_params = attention_params + mlp_params + norm_params;

        // Total params: embeddings + layers + final norm + lm_head
        let embedding_params = vocab_size * hidden_size;
        let total_params = embedding_params
            + (num_layers * layer_params)
            + hidden_size
            + (hidden_size * vocab_size);

        // Convert to GB (assuming fp16 = 2 bytes per param)
        total_params * 2.0 / (1024.0 * 1024.0 * 1024.0)
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
        let model_type = self.model_type.clone();
        let config = self.config.clone();
        let input_ids = input_ids.clone();
        let labels = labels.clone();

        // Run in spawn_blocking to avoid blocking the async runtime
        let result: std::result::Result<SftStepMetrics, Error> =
            tokio::task::spawn_blocking(move || {
                // Get model parameters ONCE at the start - reuse throughout the step
                // Each get_parameters() call clones ~70 parameter tensors (3-4GB for 1.7B model)
                let params = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.get_parameters()?
                };

                // Build loss config
                let loss_config = SftLossConfig {
                    ignore_index: Some(-100),
                    label_smoothing: config.label_smoothing,
                };

                // Compute loss and gradients
                let (loss_value, gradients) = compute_sft_loss_and_gradients(
                    &model_type,
                    &params,
                    &input_ids,
                    &labels,
                    loss_config,
                    config.gradient_checkpointing.unwrap_or(true),
                )?;

                // Check for NaN/Inf in gradients BEFORE accumulation
                // This catches numerical instability earlier than loss-only checking
                // Uses GPU-native has_nan_or_inf() to avoid transferring entire gradient tensors to CPU
                let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
                for (name, grad) in &gradients {
                    grad.eval();
                    // GPU-native check: only transfers a single boolean to CPU
                    let has_invalid = grad.has_nan_or_inf()?;
                    if has_invalid {
                        let mut state = state_arc.write().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                        })?;
                        state.nan_gradient_count += 1;
                        state.consecutive_nan_count += 1;

                        // Only do detailed CPU analysis in verbose mode (for debugging)
                        if verbose_nan {
                            let grad_data = grad.to_float32()?;
                            let nan_count = grad_data.iter().filter(|v| v.is_nan()).count();
                            let inf_count = grad_data.iter().filter(|v| v.is_infinite()).count();
                            warn!(
                                "NaN/Inf gradient detected in parameter '{}': {} NaN, {} Inf (count: {})",
                                name, nan_count, inf_count, state.nan_gradient_count
                            );
                        } else {
                            warn!(
                                "NaN/Inf gradient detected in parameter '{}', skipping step (count: {})",
                                name, state.nan_gradient_count
                            );
                        }

                        let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                        if state.consecutive_nan_count >= emergency_threshold {
                            state.needs_emergency_save = true;
                            warn!(
                                "Emergency save triggered: {} consecutive NaN gradients",
                                state.consecutive_nan_count
                            );
                        }

                        return Ok(SftStepMetrics {
                            step: state.step,
                            loss: loss_value,
                            total_tokens: 0,
                            token_accuracy: None,
                            gradients_applied: false,
                            training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                        });
                    }
                }

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
                                // FORCE EVALUATION to break lazy chain and prevent memory growth
                                // Without this, gradient_accumulation_steps=8 creates an 8-deep lazy chain
                                acc_grad.eval();
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

                        // Apply weight decay to gradients if configured
                        // Reuse `params` from start of step - no extra get_parameters() call
                        let grads_with_decay = if weight_decay > 0.0 {
                            final_grads
                                .into_iter()
                                .map(|(name, grad)| {
                                    if let Some(param) = params.get(&name) {
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

                        // Update model parameters
                        {
                            let mut model = model_arc.write().map_err(|_| {
                                Error::new(
                                    Status::GenericFailure,
                                    "Failed to acquire model write lock",
                                )
                            })?;

                            let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                                .iter()
                                .map(|(k, v)| (k.clone(), v))
                                .collect();
                            // Use apply_gradients_with_params to avoid another get_parameters() call
                            model.apply_gradients_with_params(grads_refs, lr, &params)?;
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
                // Note: If gradients were applied, we need fresh params. Otherwise reuse existing.
                let token_accuracy = if config.compute_accuracy.unwrap_or(false) {
                    // If gradients were applied, model weights changed - get fresh params
                    // Otherwise, reuse params from start of step
                    let accuracy_params = if gradients_applied {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.get_parameters()?
                    } else {
                        // Reuse params from start of step - no weights changed
                        params.clone()
                    };
                    match compute_token_accuracy(&model_type, &accuracy_params, &input_ids, &labels) {
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

        // Get params ONCE for both weight decay and apply_gradients
        let params = {
            let model = model_arc.read().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire model read lock")
            })?;
            model.get_parameters()?
        };

        // Apply weight decay to gradients if configured
        // Reuse `params` - no extra get_parameters() call
        let grads_with_decay = if weight_decay > 0.0 {
            final_grads
                .into_iter()
                .map(|(name, grad)| {
                    if let Some(param) = params.get(&name) {
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

        {
            let mut model = model_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire model write lock")
            })?;

            let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            // Use apply_gradients_with_params to avoid another get_parameters() call
            model.apply_gradients_with_params(grads_refs, lr, &params)?;
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

    /// Get the underlying Qwen3 model for checkpointing
    #[napi]
    pub fn get_model(&self) -> Result<Qwen3Model> {
        let model = self
            .model
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        match &*model {
            TrainableModelEnum::Qwen3(m) => m.clone_for_session(),
            _ => Err(Error::new(
                Status::GenericFailure,
                "get_model() only supports Qwen3. Use get_qwen35_model() or get_qwen35_moe_model() for Qwen3.5 variants.",
            )),
        }
    }

    /// Get the underlying Qwen3.5 dense model for checkpointing
    #[napi]
    pub fn get_qwen35_model(&self) -> Result<Qwen3_5Model> {
        let model = self
            .model
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        match &*model {
            TrainableModelEnum::Qwen35Dense(m) => m.clone_for_training(),
            _ => Err(Error::new(
                Status::GenericFailure,
                "get_qwen35_model() only supports Qwen3.5 dense models.",
            )),
        }
    }

    /// Get the underlying Qwen3.5 MoE model for checkpointing
    #[napi]
    pub fn get_qwen35_moe_model(&self) -> Result<Qwen3_5MoeModel> {
        let model = self
            .model
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        match &*model {
            TrainableModelEnum::Qwen35Moe(m) => m.clone_for_training(),
            _ => Err(Error::new(
                Status::GenericFailure,
                "get_qwen35_moe_model() only supports Qwen3.5 MoE models.",
            )),
        }
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
