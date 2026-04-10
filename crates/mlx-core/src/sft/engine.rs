/// SFT Training Engine - Thin Coordinator
///
/// Routes all MLX operations through the model thread. No MxArrays or model
/// state live here - only plain data crosses the thread boundary.
///
/// ## Architecture
/// ```text
/// SftTrainingEngine (NAPI thread)
///   |-- dispatch: TrainingDispatch (sends commands to model thread)
///   |-- config: SftEngineConfig
///   `-- state: EngineState (epoch counters, emergency save flag)
///
/// Model Thread:
///   |-- training_state: ModelThreadTrainingState (optimizer, grads, NaN tracking)
///   `-- train_step_sft_sync() (loss, gradients, accumulation, optimizer)
/// ```
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{info, warn};

use crate::models::qwen3::{Qwen3Cmd, Qwen3Model};
use crate::models::qwen3_5::model::{Qwen3_5Model, Qwen35Cmd};
use crate::models::qwen3_5_moe::model::{Qwen3_5MoeModel, Qwen35MoeCmd};
use crate::training_model::{ModelType, TrainStepPlainMetrics, TrainingDispatch};

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

/// Internal training state (NAPI-side only).
///
/// Gradient accumulation, optimizer, and NaN tracking now live on the model thread
/// in `ModelThreadTrainingState`. This struct only tracks epoch-level accumulators
/// and the emergency save flag.
struct EngineState {
    /// Global step counter (mirrors model thread's step)
    step: i64,
    /// Current micro-step within gradient accumulation (mirrors model thread)
    micro_step: i32,
    /// Current epoch
    epoch: i32,
    /// Epoch metrics accumulator
    epoch_loss_sum: f64,
    epoch_steps: i64,
    epoch_tokens: i64,
    /// Cumulative NaN gradient count (mirrored from model thread metrics)
    nan_gradient_count: u64,
    /// Consecutive NaN gradient count (for emergency checkpoint detection)
    consecutive_nan_count: u32,
    /// Flag indicating an emergency checkpoint should be saved
    needs_emergency_save: bool,
    /// Lifecycle generation — bumped by `reset()` / `restore_state()` so an
    /// in-flight `train_step()` can detect that its stale post-await
    /// writeback would resurrect cleared/restored state and skip the update.
    generation: u64,
    /// Terminal invalidation flag — set by `reset()` so this handle can no
    /// longer drive the model thread's training state. Any subsequent
    /// dispatch from this engine errors out. A fresh engine must be
    /// constructed to resume training.
    invalidated: bool,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            step: 0,
            micro_step: 0,
            epoch: 0,
            epoch_loss_sum: 0.0,
            epoch_steps: 0,
            epoch_tokens: 0,
            nan_gradient_count: 0,
            consecutive_nan_count: 0,
            needs_emergency_save: false,
            generation: 0,
            invalidated: false,
        }
    }
}

/// SFT Training Engine
///
/// Thin coordinator that routes all MLX operations through the model thread.
/// No MxArrays or model state live here - only plain data crosses the boundary.
#[napi]
pub struct SftTrainingEngine {
    /// Dispatch handle for sending commands to the model thread
    dispatch: TrainingDispatch,
    /// Engine configuration
    config: SftEngineConfig,
    /// Training state (epoch counters and emergency save flag)
    state: Arc<RwLock<EngineState>>,
}

#[napi]
impl SftTrainingEngine {
    /// Create a new SFT training engine from a Qwen3 model
    #[napi(constructor)]
    pub fn new(model: &Qwen3Model, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen3(model.get_config());

        info!(
            "Creating SFT training engine: {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        // Extract the cmd_sender from the model's thread
        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        // Send InitTraining to set up optimizer + state on model thread
        let grpo_config = Self::sft_config_to_grpo_config(&config);
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen3Cmd::InitTraining {
                config: Box::new(grpo_config),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen3(sender),
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new SFT training engine from a Qwen3.5 dense model
    #[napi(factory)]
    pub fn from_qwen35(model: &Qwen3_5Model, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Dense(model.config.clone());

        info!(
            "Creating SFT training engine (Qwen3.5 Dense): {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        let grpo_config = Self::sft_config_to_grpo_config(&config);
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen35Cmd::InitTraining {
                config: Box::new(grpo_config),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen35Dense(sender),
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new SFT training engine from a Qwen3.5 MoE model
    #[napi(factory)]
    pub fn from_qwen35_moe(model: &Qwen3_5MoeModel, config: SftEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Moe(model.config.clone());

        info!(
            "Creating SFT training engine (Qwen3.5 MoE): {} layers, {} hidden, lr={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            config.learning_rate.unwrap_or(2e-5)
        );

        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        let grpo_config = Self::sft_config_to_grpo_config(&config);
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen35MoeCmd::InitTraining {
                config: Box::new(grpo_config),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen35Moe(sender),
            config,
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Run a single training step
    #[napi]
    pub async fn train_step(
        &self,
        input_ids: &crate::array::MxArray,
        labels: &crate::array::MxArray,
    ) -> Result<SftStepMetrics> {
        let training_start = std::time::Instant::now();

        // Convert MxArray to plain data for crossing thread boundary
        let ids_data: Vec<i32> = input_ids.to_int32()?.to_vec();
        let ids_shape: Vec<i64> = input_ids.shape()?.to_vec();
        let labels_data: Vec<i32> = labels.to_int32()?.to_vec();
        let labels_shape: Vec<i64> = labels.shape()?.to_vec();

        let config = self.config.clone();

        // Reject dispatch from an invalidated engine, and snapshot the
        // lifecycle generation before dispatching so that if reset() or
        // restore_state() lands while we're awaiting the model thread, the
        // stale writeback below is dropped instead of resurrecting cleared
        // state.
        let start_generation = self.ensure_valid_snapshot_generation()?;

        // Dispatch to model thread
        let metrics = self
            .dispatch_train_step_sft(ids_data, ids_shape, labels_data, labels_shape, config)
            .await?;

        let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;

        // Update engine state from model thread metrics
        {
            let mut state = self.state.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            if state.generation == start_generation {
                state.step = metrics.step;
                state.epoch_steps += 1;

                // Mirror NaN tracking
                state.nan_gradient_count = metrics.nan_gradient_count;

                // Check emergency save threshold
                if !metrics.gradients_applied && metrics.nan_gradient_count > 0 {
                    state.consecutive_nan_count += 1;
                    let emergency_threshold =
                        self.config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold
                        && !state.needs_emergency_save
                    {
                        state.needs_emergency_save = true;
                        warn!(
                            "Emergency save triggered: {} consecutive steps with NaN/Inf gradients",
                            state.consecutive_nan_count
                        );
                    }
                } else {
                    state.consecutive_nan_count = 0;
                }

                // Update epoch accumulators
                state.epoch_loss_sum += metrics.loss;
                state.epoch_tokens += metrics.total_tokens as i64;
            }
        }

        Ok(SftStepMetrics {
            step: metrics.step,
            loss: metrics.loss,
            total_tokens: metrics.total_tokens,
            token_accuracy: None, // TODO: route compute_token_accuracy through model thread
            gradients_applied: metrics.gradients_applied,
            training_time_ms,
        })
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
    /// With the model thread architecture, gradient accumulation is handled
    /// on the model thread. This method is kept for API compatibility but
    /// currently logs a warning. Partial accumulation at epoch boundaries
    /// will be handled in a future update.
    #[napi]
    pub fn flush_gradients(&self) -> Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state read lock"))?;

        if state.micro_step > 0 {
            warn!(
                "flush_gradients() called with {} accumulated micro-steps. \
                 Gradient flushing is now handled on the model thread. \
                 Partial accumulation at epoch boundary will be skipped.",
                state.micro_step
            );
        }

        Ok(false)
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
    ///
    /// This is a TERMINAL operation on this handle. It drops the training
    /// state (optimizer, step counter) on the model thread so a fresh
    /// `SftTrainingEngine` can be constructed on the same model, and marks
    /// THIS handle as invalidated. Any subsequent dispatch-requiring method
    /// on this handle returns an error — callers must construct a new
    /// engine to continue training.
    #[napi]
    pub fn reset(&self) -> Result<()> {
        // Short-circuit if already reset — idempotent.
        {
            let state = self
                .state
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
            if state.invalidated {
                return Ok(());
            }
        }

        // Drop model-thread training state first (optimizer + ts.step).
        self.dispatch_reset_training_blocking()?;

        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        // Preserve-and-bump the lifecycle generation so any in-flight
        // train_step() that resumes after this reset detects the change
        // and skips its stale writeback.
        let next_generation = state.generation.wrapping_add(1);
        *state = EngineState::default();
        state.generation = next_generation;
        state.invalidated = true;
        info!("Training state reset (engine handle invalidated)");
        Ok(())
    }

    /// Restore training state (for resuming from checkpoint)
    ///
    /// Updates both the engine's read-through cache and the model thread's
    /// authoritative `ts.step`. Does NOT touch optimizer state — that is
    /// loaded via `loadOptimizerState`, which restores the AdamW bias-
    /// correction step separately.
    #[napi]
    pub fn restore_state(&self, step: i64, epoch: i32) -> Result<()> {
        self.ensure_valid()?;
        // Update the authoritative ts.step on the model thread first.
        self.dispatch_set_training_step_blocking(step)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Lock error"))?;
        state.step = step;
        state.epoch = epoch;
        // Bump the lifecycle generation so any in-flight `train_step()` with
        // a stale `start_generation` skips its post-await writeback and
        // doesn't clobber the restored step/epoch.
        state.generation = state.generation.wrapping_add(1);
        info!("Restored training state: step={}, epoch={}", step, epoch);
        Ok(())
    }

    /// Returns an error if this engine handle has been invalidated via
    /// `reset()`. Call at the top of any method that dispatches to the
    /// model thread.
    fn ensure_valid(&self) -> Result<()> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state read lock"))?;
        if state.invalidated {
            return Err(Error::new(
                Status::GenericFailure,
                "SftTrainingEngine handle has been invalidated by reset(). \
                 Construct a new engine to continue training.",
            ));
        }
        Ok(())
    }

    /// Atomically check that this engine has not been invalidated and
    /// snapshot the current lifecycle generation. Used by `train_step()` to
    /// gate its post-await writeback.
    fn ensure_valid_snapshot_generation(&self) -> Result<u64> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state read lock"))?;
        if state.invalidated {
            return Err(Error::new(
                Status::GenericFailure,
                "SftTrainingEngine handle has been invalidated by reset(). \
                 Construct a new engine to continue training.",
            ));
        }
        Ok(state.generation)
    }

    /// Get the underlying Qwen3 model for checkpointing
    ///
    /// NOTE: With the model thread architecture, direct model access is no longer
    /// supported. Use save_checkpoint() on the model directly instead.
    #[napi]
    pub fn get_model(&self) -> Result<Qwen3Model> {
        Err(Error::new(
            Status::GenericFailure,
            "get_model() is no longer supported with the model thread architecture. \
             Use model.saveModel(path) or the SaveCheckpoint command instead.",
        ))
    }

    /// Get the underlying Qwen3.5 dense model for checkpointing
    ///
    /// NOTE: With the model thread architecture, direct model access is no longer
    /// supported. Use save_checkpoint() on the model directly instead.
    #[napi]
    pub fn get_qwen35_model(&self) -> Result<Qwen3_5Model> {
        Err(Error::new(
            Status::GenericFailure,
            "get_qwen35_model() is no longer supported with the model thread architecture. \
             Use model.saveModel(path) or the SaveCheckpoint command instead.",
        ))
    }

    /// Get the underlying Qwen3.5 MoE model for checkpointing
    ///
    /// NOTE: With the model thread architecture, direct model access is no longer
    /// supported. Use save_checkpoint() on the model directly instead.
    #[napi]
    pub fn get_qwen35_moe_model(&self) -> Result<Qwen3_5MoeModel> {
        Err(Error::new(
            Status::GenericFailure,
            "get_qwen35_moe_model() is no longer supported with the model thread architecture. \
             Use model.saveModel(path) or the SaveCheckpoint command instead.",
        ))
    }
}

// =============================================================================
// Dispatch helper methods (private, not exposed to NAPI)
// =============================================================================

impl SftTrainingEngine {
    /// Send TrainStepSFT command and await metrics.
    async fn dispatch_train_step_sft(
        &self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: SftEngineConfig,
    ) -> Result<TrainStepPlainMetrics> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::TrainStepSFT {
                        input_ids,
                        input_shape,
                        labels,
                        labels_shape,
                        config,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::TrainStepSFT {
                        input_ids,
                        input_shape,
                        labels,
                        labels_shape,
                        config,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::TrainStepSFT {
                        input_ids,
                        input_shape,
                        labels,
                        labels_shape,
                        config,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Send ResetTraining command and block until complete.
    ///
    /// Drops the training state (optimizer + step counter) on the model
    /// thread. Blocking because it's invoked from the sync `reset()` NAPI
    /// method.
    fn dispatch_reset_training_blocking(&self) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::ResetTraining { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::ResetTraining { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::ResetTraining { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Send SetTrainingStep command and block until complete.
    ///
    /// Plumbs the restored step to the model thread's authoritative
    /// `ts.step` so subsequent `train_step_sft_sync` increments from the
    /// correct value. Blocking because it's invoked from the sync
    /// `restore_state()` NAPI method.
    fn dispatch_set_training_step_blocking(&self, step: i64) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::SetTrainingStep { step, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::SetTrainingStep { step, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::SetTrainingStep { step, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Convert SftEngineConfig to GRPOEngineConfig for InitTraining command.
    ///
    /// The InitTraining command was designed for GRPO, but the common fields
    /// (optimizer, gradient config, NaN tracking) are shared with SFT.
    /// GRPO-specific fields are set to defaults.
    fn sft_config_to_grpo_config(
        config: &SftEngineConfig,
    ) -> crate::grpo::engine::GRPOEngineConfig {
        crate::grpo::engine::GRPOEngineConfig {
            learning_rate: config.learning_rate,
            gradient_accumulation_steps: config.gradient_accumulation_steps,
            gradient_clip_norm: config.gradient_clip_norm,
            gradient_clip_value: config.gradient_clip_value,
            max_nan_gradients: config.max_nan_gradients,
            emergency_save_threshold: config.emergency_save_threshold,
            verbose_nan_detection: config.verbose_nan_detection,
            gradient_checkpointing: config.gradient_checkpointing,
            weight_decay: config.weight_decay,
            // GRPO-specific fields at defaults
            group_size: Some(4),
            clip_epsilon: Some(0.2),
            kl_coef: Some(0.0),
            loss_type: Some("grpo".to_string()),
            max_completion_length: Some(256),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: None,
            repetition_penalty: Some(1.1),
            presence_penalty: None,
            frequency_penalty: None,
            enable_thinking: Some(true),
            tools: None,
            lm_head_chunk_size: Some(2),
            forward_chunk_size: None,
            vocab_chunk_size: Some(65536),
            use_parallel_batch_generation: Some(false),
            optimizer_type: Some("adamw".to_string()),
            adamw_beta1: Some(0.9),
            adamw_beta2: Some(0.999),
            adamw_eps: Some(1e-8),
        }
    }
}
