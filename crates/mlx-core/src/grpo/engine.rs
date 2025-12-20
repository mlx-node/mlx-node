/// GRPO Training Engine - Rust-native Training Loop
///
/// This module provides a complete GRPO training engine that runs entirely in Rust,
/// eliminating FFI overhead and enabling potential mx.compile() optimization.
///
/// ## Key Features
/// - Complete training loop in Rust (generate → score → train)
/// - Built-in reward functions (no FFI for common patterns)
/// - Optional JS callback for custom rewards
/// - Gradient accumulation and memory management
/// - Comprehensive logging and metrics
///
/// ## Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────┐
/// │  GRPOTrainingEngine                                 │
/// │  ├── model: Qwen3Model (with parameters)            │
/// │  ├── config: Engine configuration                   │
/// │  ├── reward_registry: Built-in + JS rewards         │
/// │  └── state: Training progress tracking              │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// ## Usage
/// ```ignore
/// const model = await Qwen3Model.loadPretrained(modelPath);
/// const engine = new GRPOTrainingEngine(model, config);
/// engine.registerBuiltinReward({ rewardType: 'ToolUse', allowedTools: ['search'] });
///
/// for (const batch of dataset) {
///   const metrics = await engine.trainStep(batch.prompts);
///   console.log(metrics);
/// }
/// ```
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::{
    MxArray, get_active_memory, get_cache_memory, heavy_cleanup, synchronize_and_clear_cache,
};
use crate::grpo::advantages::compute_advantages;
use crate::grpo::autograd::compute_loss_and_gradients_autograd;
use crate::grpo::loss::GRPOLossConfig;
use crate::grpo::rewards::{
    BuiltinRewardConfig, JsonSchemaReward, LengthReward, RewardRegistry, ToolUseReward,
    XMLFormatReward,
};
use crate::models::qwen3::{GenerationConfig, Qwen3Config, Qwen3Model};
use crate::optimizers::GradientUtils;
use crate::tokenizer::ChatMessage;
use crate::tools::build_reward_outputs;

/// Configuration for the GRPO training engine
#[napi(object)]
#[derive(Clone)]
pub struct GRPOEngineConfig {
    // === Training hyperparameters ===
    /// Learning rate (default: 1e-6)
    pub learning_rate: Option<f64>,
    /// Gradient accumulation steps (default: 1)
    pub gradient_accumulation_steps: Option<i32>,
    /// Maximum gradient norm for clipping (default: 1.0)
    pub gradient_clip_norm: Option<f64>,
    /// Maximum gradient value for element-wise clipping (default: 1.0)
    /// This clamps individual gradient elements to [-value, value]
    pub gradient_clip_value: Option<f64>,

    // === GRPO hyperparameters ===
    /// Number of completions per prompt (default: 4)
    pub group_size: Option<i32>,
    /// PPO clipping epsilon (default: 0.2)
    pub clip_epsilon: Option<f64>,
    /// KL divergence coefficient (default: 0.0)
    pub kl_coef: Option<f64>,
    /// Loss type: "grpo", "dapo", "dr_grpo", "bnpo" (default: "grpo")
    pub loss_type: Option<String>,

    // === Generation parameters ===
    /// Maximum tokens to generate (default: 256)
    pub max_new_tokens: Option<i32>,
    /// Maximum completion length for training/autograd (default: 1024)
    /// Completions longer than this are truncated before computing gradients.
    /// Separate from max_new_tokens to allow generating long outputs
    /// while limiting memory usage during training.
    pub max_completion_length_for_training: Option<i32>,
    /// Sampling temperature (default: 0.8)
    pub temperature: Option<f64>,
    /// Top-p (nucleus) sampling (default: 0.95)
    pub top_p: Option<f64>,
    /// Top-k sampling (optional)
    pub top_k: Option<i32>,
    /// Repetition penalty (default: 1.1)
    pub repetition_penalty: Option<f64>,

    // === Memory management parameters ===
    /// Steps between heavy cleanup to prevent GPU timeout (default: 25)
    /// Heavy cleanup forces complete GPU drain including peak memory reset
    pub heavy_cleanup_interval: Option<i32>,
    /// Memory threshold for triggering cleanup (bytes). Default: 80% of system memory.
    /// When memory usage exceeds this threshold, cleanup is triggered regardless of step interval.
    pub memory_cleanup_threshold: Option<f64>,

    // === NaN gradient protection ===
    /// Maximum allowed NaN gradient occurrences before stopping training (default: 100)
    /// When exceeded, training will stop with an error to prevent model corruption.
    pub max_nan_gradients: Option<i64>,
    /// Consecutive NaN gradients that trigger emergency checkpoint (default: 5)
    /// When reached, the needs_emergency_save flag is set for the TypeScript layer.
    pub emergency_save_threshold: Option<i32>,
}

impl Default for GRPOEngineConfig {
    fn default() -> Self {
        Self {
            learning_rate: Some(1e-6),
            gradient_accumulation_steps: Some(1),
            gradient_clip_norm: Some(1.0),
            gradient_clip_value: Some(1.0),
            group_size: Some(4),
            clip_epsilon: Some(0.2),
            kl_coef: Some(0.0),
            loss_type: Some("grpo".to_string()),
            max_new_tokens: Some(256),
            max_completion_length_for_training: Some(1024),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: None,
            repetition_penalty: Some(1.1),
            heavy_cleanup_interval: Some(25),
            memory_cleanup_threshold: None, // Default: 80% of system memory at runtime
            max_nan_gradients: Some(100),
            emergency_save_threshold: Some(5),
        }
    }
}

/// Metrics from a single training step
#[napi(object)]
#[derive(Clone)]
pub struct EngineStepMetrics {
    /// Current step number
    pub step: i64,
    /// GRPO loss value
    pub loss: f64,
    /// Mean reward across completions
    pub mean_reward: f64,
    /// Standard deviation of rewards
    pub std_reward: f64,
    /// Mean advantage value
    pub mean_advantage: f64,
    /// Total tokens generated this step
    pub total_tokens: i32,
    /// Whether gradients were applied
    pub gradients_applied: bool,
    /// Time for generation (ms)
    pub generation_time_ms: f64,
    /// Time for training (ms)
    pub training_time_ms: f64,
}

/// Result from generate_batch_for_training with all data needed for training
#[napi(object)]
#[derive(Clone)]
pub struct GenerateBatchResult {
    /// Generated completion texts
    pub completion_texts: Vec<String>,
    /// Completion token IDs (flattened, concatenated)
    pub completion_tokens: Vec<i64>,
    /// Completion log probabilities (flattened, concatenated)
    pub completion_logprobs: Vec<f64>,
    /// Lengths of each completion (for reconstruction)
    pub completion_lengths: Vec<i32>,
    /// Finish reasons for each completion ("eos", "length", or "repetition")
    pub finish_reasons: Vec<String>,
}

/// Metrics from a training epoch
#[napi(object)]
#[derive(Clone)]
pub struct EngineEpochMetrics {
    /// Epoch number
    pub epoch: i32,
    /// Average loss for the epoch
    pub avg_loss: f64,
    /// Average reward for the epoch
    pub avg_reward: f64,
    /// Total steps in the epoch
    pub total_steps: i64,
    /// Total tokens processed
    pub total_tokens: i64,
    /// Time for the epoch (seconds)
    pub epoch_time_secs: f64,
}

/// Result from train_step_auto including metrics, completions, and rewards
#[napi(object)]
#[derive(Clone)]
pub struct TrainStepResult {
    /// Training metrics
    pub metrics: EngineStepMetrics,
    /// Generated completion texts (for TUI logging)
    pub completions: Vec<String>,
    /// Computed reward values (for TUI logging)
    pub rewards: Vec<f64>,
}

/// Internal training state
struct EngineState {
    /// Accumulated gradients
    accumulated_gradients: Option<HashMap<String, MxArray>>,
    /// Current micro-step within gradient accumulation
    micro_step: i32,
    /// Global step counter
    step: i64,
    /// Current epoch
    epoch: i32,
    /// Epoch metrics accumulator
    epoch_loss_sum: f64,
    epoch_reward_sum: f64,
    epoch_steps: i64,
    epoch_tokens: i64,
    /// Last step when heavy cleanup was performed
    last_heavy_cleanup_step: i64,
    /// Cumulative NaN gradient count across training
    nan_gradient_count: u64,
    /// Consecutive NaN gradient count (for emergency checkpoint detection)
    consecutive_nan_count: u32,
    /// Flag indicating an emergency checkpoint should be saved
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
            epoch_reward_sum: 0.0,
            epoch_steps: 0,
            epoch_tokens: 0,
            last_heavy_cleanup_step: 0,
            nan_gradient_count: 0,
            consecutive_nan_count: 0,
            needs_emergency_save: false,
        }
    }
}

/// GRPO Training Engine
///
/// Complete training engine that runs entirely in Rust.
#[napi]
pub struct GRPOTrainingEngine {
    /// The model being trained
    model: Arc<RwLock<Qwen3Model>>,
    /// Model configuration (for functional forward pass)
    model_config: Qwen3Config,
    /// Engine configuration
    config: GRPOEngineConfig,
    /// Reward registry (built-in rewards)
    reward_registry: RewardRegistry,
    /// Training state
    state: Arc<RwLock<EngineState>>,
}

#[napi]
impl GRPOTrainingEngine {
    /// Create a new training engine from an existing model
    ///
    /// # Arguments
    /// * `model` - The Qwen3 model to train (will be cloned internally)
    /// * `config` - Engine configuration
    #[napi(constructor)]
    pub fn new(model: &Qwen3Model, config: GRPOEngineConfig) -> Result<Self> {
        let model_config = model.get_config();

        info!(
            "Creating training engine: {} layers, {} hidden, eos_token_id={}, pad_token_id={}",
            model_config.num_layers,
            model_config.hidden_size,
            model_config.eos_token_id,
            model_config.pad_token_id
        );

        Ok(Self {
            model: Arc::new(RwLock::new(model.clone_for_session()?)),
            model_config,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Register a built-in reward function
    #[napi]
    pub fn register_builtin_reward(&mut self, config: BuiltinRewardConfig) -> Result<()> {
        let weight = config.weight.unwrap_or(1.0);

        match config.reward_type {
            crate::grpo::rewards::BuiltinRewardType::ToolUse => {
                let tools: Vec<&str> = config
                    .allowed_tools
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_else(|| vec!["search", "calculate", "code"]);
                let required = config.required.unwrap_or(true);

                self.reward_registry.register_builtin(
                    "tool_use",
                    ToolUseReward::new(&tools, required),
                    weight,
                );
                info!("Registered tool_use reward with tools: {:?}", tools);
            }
            crate::grpo::rewards::BuiltinRewardType::XmlFormat => {
                let tags: Vec<&str> = config
                    .required_tags
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_else(|| vec!["thinking", "answer"]);

                self.reward_registry.register_builtin(
                    "xml_format",
                    XMLFormatReward::new(&tags),
                    weight,
                );
                info!("Registered xml_format reward with tags: {:?}", tags);
            }
            crate::grpo::rewards::BuiltinRewardType::Length => {
                let min = config.min_length.unwrap_or(50) as usize;
                let max = config.max_length.unwrap_or(500) as usize;
                let use_chars = config.use_chars.unwrap_or(true);

                self.reward_registry.register_builtin(
                    "length",
                    LengthReward::new(min, max, use_chars),
                    weight,
                );
                info!(
                    "Registered length reward: min={}, max={}, chars={}",
                    min, max, use_chars
                );
            }
            crate::grpo::rewards::BuiltinRewardType::JsonSchema => {
                let fields: Vec<&str> = config
                    .required_fields
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();

                self.reward_registry.register_builtin(
                    "json_schema",
                    JsonSchemaReward::new(&fields),
                    weight,
                );
                info!("Registered json_schema reward with fields: {:?}", fields);
            }
        }

        Ok(())
    }

    /// Run a training step with provided rewards
    ///
    /// This method performs the complete training cycle:
    /// 1. Generate completions for each prompt (G times per prompt)
    /// 2. Use provided rewards to compute advantages
    /// 3. Compute GRPO loss and gradients
    /// 4. Apply gradients (respecting accumulation steps)
    ///
    /// # Arguments
    /// * `prompts` - Array of chat conversations to use as prompts
    /// * `rewards` - Reward values for each completion (num_prompts * group_size)
    ///
    /// # Returns
    /// * Training step metrics
    #[napi]
    pub async fn train_step(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        rewards: Vec<f64>,
    ) -> Result<EngineStepMetrics> {
        let num_prompts = prompts.len();
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let expected_rewards = num_prompts * group_size;

        if rewards.len() != expected_rewards {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Expected {} rewards ({}×{}), got {}",
                    expected_rewards,
                    num_prompts,
                    group_size,
                    rewards.len()
                ),
            ));
        }

        let generation_start = std::time::Instant::now();

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let model_config = self.model_config.clone();
        let config = self.config.clone();

        // Build generation config - use model's eos_token_id explicitly
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_config.eos_token_id),
            return_logprobs: Some(true),
        };

        // Run the entire training step in spawn_blocking
        let metrics = napi::bindgen_prelude::spawn_blocking(move || {
            // === Phase 1: Generate completions ===
            let mut prompt_tokens_all: Vec<MxArray> = Vec::with_capacity(num_prompts);
            let mut completion_tokens_all: Vec<MxArray> =
                Vec::with_capacity(num_prompts * group_size);
            let mut completion_logprobs_all: Vec<MxArray> =
                Vec::with_capacity(num_prompts * group_size);
            let mut token_counts_all: Vec<i32> = Vec::with_capacity(num_prompts * group_size);

            for prompt_messages in prompts {
                // Tokenize prompt
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(&prompt_messages, Some(true))?
                };

                let prompt_array =
                    MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
                prompt_tokens_all.push(prompt_array.squeeze(Some(&[0]))?);

                // Generate G completions
                for _g in 0..group_size {
                    let result = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.generate_for_training_sync(&prompt_array, Some(gen_config.clone()))?
                    };

                    completion_tokens_all.push(result.tokens.clone());
                    completion_logprobs_all.push(result.logprobs.clone());
                    token_counts_all.push(result.num_tokens as i32);

                    // CRITICAL: Clear KV cache and intermediate tensors after each completion
                    // Without this, memory accumulates O(group_size × completion_length)
                    synchronize_and_clear_cache();
                }
            }

            let generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;

            // Sync and clear GPU memory after generation phase to reduce fragmentation
            // This releases intermediate tensors from generation before building training graph
            synchronize_and_clear_cache();

            let training_start = std::time::Instant::now();

            // === Phase 2: Compute loss and gradients ===
            let loss_config = GRPOLossConfig {
                epsilon_low: config.clip_epsilon.unwrap_or(0.2),
                epsilon_high: None,
                beta: config.kl_coef.unwrap_or(0.0),
                loss_type: config
                    .loss_type
                    .clone()
                    .unwrap_or_else(|| "grpo".to_string()),
                importance_sampling_level: "token".to_string(),
                max_completion_length: config.max_completion_length_for_training.or(config.max_new_tokens).map(|n| n as i64),
                num_items_in_batch: Some(
                    (num_prompts * config.group_size.unwrap_or(4) as usize) as f64,
                ),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens_all.iter().collect();
            let completion_refs: Vec<&MxArray> = completion_tokens_all.iter().collect();
            let logprob_refs: Vec<&MxArray> = completion_logprobs_all.iter().collect();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_config,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards,
                config.group_size.unwrap_or(4),
                loss_config,
            )?;

            // Check for NaN
            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                let total_tokens: i32 = token_counts_all.iter().sum();

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.step += 1;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            // Step 1: Validate ALL gradients first - if ANY has NaN/Inf, skip entire step
            // This prevents partial gradient application which can degrade model weights
            for (name, grad) in gradients.iter() {
                grad.eval();
                let data = grad.to_float32()?;
                let invalid_count = data
                    .iter()
                    .filter(|v| v.is_nan() || v.is_infinite())
                    .count();
                if invalid_count > 0 {
                    warn!(
                        "Gradient '{}' contains {} invalid values (NaN/Inf) - SKIPPING ENTIRE STEP to prevent model corruption",
                        name, invalid_count
                    );

                    // Update NaN tracking
                    let mut state = state_arc.write().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                    })?;
                    state.nan_gradient_count += 1;
                    state.consecutive_nan_count += 1;
                    let max_nan = config.max_nan_gradients.unwrap_or(100) as u64;
                    warn!(
                        "NaN gradient count: {} / {} (consecutive: {})",
                        state.nan_gradient_count, max_nan, state.consecutive_nan_count
                    );

                    // Check if we've exceeded max NaN threshold
                    if state.nan_gradient_count >= max_nan {
                        return Err(Error::new(
                            Status::GenericFailure,
                            format!(
                                "Training stopped: exceeded maximum NaN gradient count ({}/{})",
                                state.nan_gradient_count, max_nan
                            ),
                        ));
                    }

                    // Check emergency save threshold (5 consecutive NaNs)
                    let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold {
                        state.needs_emergency_save = true;
                        warn!(
                            "Emergency save triggered: {} consecutive NaN gradients",
                            state.consecutive_nan_count
                        );
                    }

                    state.step += 1;
                    let current_step = state.step;
                    drop(state);

                    // Compute metrics for reporting
                    let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                    let total_tokens: i32 = token_counts_all.iter().sum();

                    synchronize_and_clear_cache();

                    // Return early WITHOUT applying any gradients
                    return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                        step: current_step,
                        loss: loss_value,
                        mean_reward,
                        std_reward,
                        mean_advantage: 0.0,
                        total_tokens,
                        gradients_applied: false,
                        generation_time_ms,
                        training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                    });
                }
            }

            // Step 2: Clamp gradient values to prevent extreme values
            // This happens BEFORE norm clipping, as extreme values break norm computation
            let grad_clip_value = config.gradient_clip_value.unwrap_or(1.0);
            let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();

            for (name, grad) in gradients.iter() {
                // Clamp to reasonable range
                let clamped = grad.clip(Some(-grad_clip_value), Some(grad_clip_value))?;
                clamped.eval();
                clamped_gradients.insert(name.clone(), clamped);
            }

            // Step 3: Apply gradient norm clipping
            let gradients = if let Some(max_norm) = config.gradient_clip_norm {
                let grad_refs: HashMap<String, &MxArray> =
                    clamped_gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
                GradientUtils::clip_grad_norm(grad_refs, max_norm)?
            } else {
                clamped_gradients
            };

            // === Phase 3: Accumulate and apply gradients ===
            // Reset consecutive NaN count since we're applying gradients
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            state.consecutive_nan_count = 0;

            accumulate_gradients(&mut state, gradients)?;
            state.micro_step += 1;

            let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
            let gradients_applied = if state.micro_step >= grad_acc_steps {
                let grads = state.accumulated_gradients.take().ok_or_else(|| {
                    Error::new(Status::GenericFailure, "No accumulated gradients")
                })?;

                let lr = config.learning_rate.unwrap_or(1e-6) / grad_acc_steps as f64;

                // Release state lock, acquire model lock
                drop(state);

                let mut model_mut = model_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model write lock")
                })?;

                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                model_mut.apply_gradients(grads_refs, lr)?;

                debug!("Applied gradients with lr: {}", lr);

                // Re-acquire state lock
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.accumulated_gradients = None;
                state.micro_step = 0;
                state.step += 1;
                state.epoch_steps += 1;

                true
            } else {
                state.step += 1;
                state.epoch_steps += 1;
                // CRITICAL: Release state lock to prevent deadlock when
                // re-acquiring for epoch accumulators. The if-branch drops
                // its lock explicitly, but this else-branch was missing it.
                drop(state);
                false
            };

            // Compute metrics
            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: i32 = token_counts_all.iter().sum();

            // Update epoch accumulators
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.epoch_loss_sum += loss_value;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }

            // Compute mean advantage
            let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
            let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
            let advantages = compute_advantages(
                &rewards_array,
                config.group_size.unwrap_or(4),
                "group".to_string(),
            )?;
            let adv_data = advantages.to_float32()?;
            let mean_advantage =
                adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len() as f64;

            synchronize_and_clear_cache();

            // Periodic heavy cleanup to prevent GPU timeout in long-running training
            let heavy_cleanup_interval = config.heavy_cleanup_interval.unwrap_or(25);
            let memory_cleanup_threshold = config.memory_cleanup_threshold.unwrap_or(0.0);
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                maybe_heavy_cleanup(&mut state, heavy_cleanup_interval, memory_cleanup_threshold);
            }

            let step = state_arc
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?
                .step;

            Ok(EngineStepMetrics {
                step,
                loss: loss_value,
                mean_reward,
                std_reward,
                mean_advantage,
                total_tokens,
                gradients_applied,
                generation_time_ms,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error: {}", e),
            )
        })??;

        Ok(metrics)
    }

    /// Generate completions without training
    ///
    /// Use this to generate completions for scoring by external reward functions.
    /// Returns completion texts along with the internal token data needed for training.
    #[napi]
    pub async fn generate_batch(&self, prompts: Vec<Vec<ChatMessage>>) -> Result<Vec<String>> {
        let result = self.generate_batch_for_training(prompts).await?;
        Ok(result.completion_texts)
    }

    /// Generate completions with all data needed for training
    ///
    /// Returns completion texts, tokens, log probabilities, and lengths.
    /// Use this when you need to score completions externally and then train.
    #[napi]
    pub async fn generate_batch_for_training(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
    ) -> Result<GenerateBatchResult> {
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let num_prompts = prompts.len();

        let model_arc = Arc::clone(&self.model);
        let config = self.config.clone();
        let model_config = self.model_config.clone();

        // Build generation config - use model's eos_token_id explicitly
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_config.eos_token_id),
            return_logprobs: Some(true),
        };

        let result = napi::bindgen_prelude::spawn_blocking(move || {
            let num_completions = num_prompts * group_size;
            let max_tokens = gen_config.max_new_tokens.unwrap_or(256) as usize;
            // Pre-allocate with expected capacity: completions * max tokens each
            // This prevents exponential reallocation during extend() calls
            let mut completion_texts: Vec<String> = Vec::with_capacity(num_completions);
            let mut all_tokens: Vec<i64> = Vec::with_capacity(num_completions * max_tokens);
            let mut all_logprobs: Vec<f64> = Vec::with_capacity(num_completions * max_tokens);
            let mut completion_lengths: Vec<i32> = Vec::with_capacity(num_completions);
            let mut finish_reasons: Vec<String> = Vec::with_capacity(num_completions);

            for prompt_messages in prompts {
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(&prompt_messages, Some(true))?
                };

                let prompt_array =
                    MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;

                for _g in 0..group_size {
                    let result = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.generate_for_training_sync(&prompt_array, Some(gen_config.clone()))?
                    };

                    let text = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.decode_tokens_sync(&result.tokens)?
                    };
                    completion_texts.push(text);
                    finish_reasons.push(result.finish_reason.clone());

                    // Extract token IDs and logprobs
                    let tokens = result.tokens.to_int32()?;
                    let logprobs = result.logprobs.to_float32()?;

                    completion_lengths.push(tokens.len() as i32);
                    all_tokens.extend(tokens.iter().map(|&x| x as i64));
                    all_logprobs.extend(logprobs.iter().map(|&x| x as f64));

                    // CRITICAL: Clear KV cache and intermediate tensors after each completion
                    // Without this, memory accumulates O(group_size × completion_length)
                    synchronize_and_clear_cache();
                }
            }

            synchronize_and_clear_cache();
            Ok::<GenerateBatchResult, Error>(GenerateBatchResult {
                completion_texts,
                completion_tokens: all_tokens,
                completion_logprobs: all_logprobs,
                completion_lengths,
                finish_reasons,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error: {}", e),
            )
        })??;

        Ok(result)
    }

    /// Run a training step with pre-generated completions
    ///
    /// This method performs training using pre-generated completions,
    /// eliminating the double-generation issue.
    ///
    /// # Arguments
    /// * `prompts` - Array of chat conversations to use as prompts
    /// * `rewards` - Reward values for each completion (num_prompts * group_size)
    /// * `generation_result` - Pre-generated completion data from generate_batch_for_training
    ///
    /// # Returns
    /// * Training step metrics
    #[napi]
    pub async fn train_step_with_generations(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        rewards: Vec<f64>,
        generation_result: GenerateBatchResult,
    ) -> Result<EngineStepMetrics> {
        let num_prompts = prompts.len();
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let expected_rewards = num_prompts * group_size;

        if rewards.len() != expected_rewards {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Expected {} rewards ({}×{}), got {}",
                    expected_rewards,
                    num_prompts,
                    group_size,
                    rewards.len()
                ),
            ));
        }

        if generation_result.completion_lengths.len() != expected_rewards {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Expected {} completions ({}×{}), got {}",
                    expected_rewards,
                    num_prompts,
                    group_size,
                    generation_result.completion_lengths.len()
                ),
            ));
        }

        let training_start = std::time::Instant::now();

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let model_config = self.model_config.clone();
        let config = self.config.clone();

        // Run the training step in spawn_blocking
        let metrics = napi::bindgen_prelude::spawn_blocking(move || {
            // === Phase 1: Tokenize prompts and reconstruct completion arrays ===
            let mut prompt_tokens_all: Vec<MxArray> = Vec::with_capacity(num_prompts);

            for prompt_messages in prompts {
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(&prompt_messages, Some(true))?
                };

                let prompt_array =
                    MxArray::from_uint32(&prompt_token_ids, &[prompt_token_ids.len() as i64])?;
                prompt_tokens_all.push(prompt_array);
            }

            // Reconstruct completion token/logprob arrays from flattened data
            let mut completion_tokens_all: Vec<MxArray> = Vec::with_capacity(expected_rewards);
            let mut completion_logprobs_all: Vec<MxArray> = Vec::with_capacity(expected_rewards);
            let mut token_counts_all: Vec<i32> = Vec::with_capacity(expected_rewards);

            let mut offset = 0usize;
            for &length in &generation_result.completion_lengths {
                let len = length as usize;
                let end = offset + len;

                let tokens = &generation_result.completion_tokens[offset..end];
                let logprobs: Vec<f32> = generation_result.completion_logprobs[offset..end]
                    .iter()
                    .map(|&x| x as f32)
                    .collect();

                let tokens_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
                completion_tokens_all.push(MxArray::from_int32(&tokens_i32, &[len as i64])?);
                completion_logprobs_all.push(MxArray::from_float32(&logprobs, &[len as i64])?);
                token_counts_all.push(length);

                offset = end;
            }

            // Sync and clear GPU memory after Phase 1 to reduce fragmentation
            // This releases intermediate tensors before building training graph
            synchronize_and_clear_cache();

            // === Phase 2: Compute loss and gradients ===
            let loss_config = GRPOLossConfig {
                epsilon_low: config.clip_epsilon.unwrap_or(0.2),
                epsilon_high: None,
                beta: config.kl_coef.unwrap_or(0.0),
                loss_type: config
                    .loss_type
                    .clone()
                    .unwrap_or_else(|| "grpo".to_string()),
                importance_sampling_level: "token".to_string(),
                max_completion_length: config.max_completion_length_for_training.or(config.max_new_tokens).map(|n| n as i64),
                num_items_in_batch: Some(expected_rewards as f64),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens_all.iter().collect();
            let completion_refs: Vec<&MxArray> = completion_tokens_all.iter().collect();
            let logprob_refs: Vec<&MxArray> = completion_logprobs_all.iter().collect();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_config,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards,
                config.group_size.unwrap_or(4),
                loss_config,
            )?;

            // Check for NaN
            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                let total_tokens: i32 = token_counts_all.iter().sum();

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.step += 1;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms: 0.0, // Not measured here, was done separately
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            // Step 1: Clamp gradient values to prevent extreme values
            // This happens BEFORE norm clipping, as Inf values break norm computation
            let grad_clip_value = config.gradient_clip_value.unwrap_or(1.0);
            let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
            let mut has_invalid_grad = false;

            for (name, grad) in gradients.iter() {
                grad.eval();
                let data = grad.to_float32()?;
                let invalid_count = data
                    .iter()
                    .filter(|v| v.is_nan() || v.is_infinite())
                    .count();

                // Check for NaN/Inf - if present, replace with zeros (skip this gradient)
                if invalid_count > 0 {
                    warn!(
                        "Gradient '{}' contains {} invalid values (NaN/Inf), replacing with zeros",
                        name, invalid_count
                    );
                    has_invalid_grad = true;
                    // Create a zero gradient with the same shape
                    let shape = grad.shape()?;
                    let zero_grad = MxArray::zeros(&shape, None)?;
                    clamped_gradients.insert(name.clone(), zero_grad);
                    continue; // Continue with other gradients instead of breaking
                }

                // Clamp to reasonable range (lazy - let MLX fuse operations)
                let clamped = grad.clip(Some(-grad_clip_value), Some(grad_clip_value))?;
                clamped_gradients.insert(name.clone(), clamped);
            }

            // Step 2: Apply gradient norm clipping
            let gradients = if !has_invalid_grad {
                if let Some(max_norm) = config.gradient_clip_norm {
                    let grad_refs: HashMap<String, &MxArray> =
                        clamped_gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
                    GradientUtils::clip_grad_norm(grad_refs, max_norm)?
                } else {
                    clamped_gradients
                }
            } else {
                clamped_gradients
            };

            if has_invalid_grad {
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                let total_tokens: i32 = token_counts_all.iter().sum();

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.step += 1;

                // Track NaN gradient occurrences
                state.nan_gradient_count += 1;
                state.consecutive_nan_count += 1;

                let max_nan = config.max_nan_gradients.unwrap_or(100) as u64;
                let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;

                // Check if we should trigger emergency checkpoint
                if state.consecutive_nan_count >= emergency_threshold && !state.needs_emergency_save {
                    warn!(
                        "Consecutive NaN gradients ({}) reached threshold ({}), flagging for emergency checkpoint",
                        state.consecutive_nan_count, emergency_threshold
                    );
                    state.needs_emergency_save = true;
                }

                // Check if we've exceeded maximum NaN gradient count
                if state.nan_gradient_count > max_nan {
                    return Err(Error::new(
                        Status::GenericFailure,
                        format!(
                            "Training stopped: {} NaN gradients exceeded threshold of {}. \
                            Model weights may be corrupted. Consider using an earlier checkpoint or reducing learning rate.",
                            state.nan_gradient_count, max_nan
                        ),
                    ));
                }

                warn!(
                    "NaN gradient count: {} / {} (consecutive: {})",
                    state.nan_gradient_count, max_nan, state.consecutive_nan_count
                );

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms: 0.0,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            // === Phase 3: Accumulate and apply gradients ===
            // Reset consecutive NaN count on successful gradient computation
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            state.consecutive_nan_count = 0;

            accumulate_gradients(&mut state, gradients)?;
            state.micro_step += 1;

            let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
            let gradients_applied = if state.micro_step >= grad_acc_steps {
                let grads = state.accumulated_gradients.take().ok_or_else(|| {
                    Error::new(Status::GenericFailure, "No accumulated gradients")
                })?;

                let lr = config.learning_rate.unwrap_or(1e-6) / grad_acc_steps as f64;

                // Release state lock, acquire model lock
                drop(state);

                let mut model_mut = model_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model write lock")
                })?;

                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                model_mut.apply_gradients(grads_refs, lr)?;

                debug!("Applied gradients with lr: {}", lr);

                // Re-acquire state lock
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.accumulated_gradients = None;
                state.micro_step = 0;
                state.step += 1;
                state.epoch_steps += 1;

                true
            } else {
                state.step += 1;
                state.epoch_steps += 1;
                // CRITICAL: Release state lock to prevent deadlock when
                // re-acquiring for epoch accumulators. The if-branch drops
                // its lock explicitly, but this else-branch was missing it.
                drop(state);
                false
            };

            // Compute metrics
            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: i32 = token_counts_all.iter().sum();

            // Update epoch accumulators
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.epoch_loss_sum += loss_value;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }

            // Compute mean advantage
            let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
            let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
            let advantages = compute_advantages(
                &rewards_array,
                config.group_size.unwrap_or(4),
                "group".to_string(),
            )?;
            let adv_data = advantages.to_float32()?;
            let mean_advantage =
                adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len() as f64;

            synchronize_and_clear_cache();

            // Periodic heavy cleanup to prevent GPU timeout in long-running training
            let heavy_cleanup_interval = config.heavy_cleanup_interval.unwrap_or(25);
            let memory_cleanup_threshold = config.memory_cleanup_threshold.unwrap_or(0.0);
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                maybe_heavy_cleanup(&mut state, heavy_cleanup_interval, memory_cleanup_threshold);
            }

            let step = state_arc
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?
                .step;

            Ok(EngineStepMetrics {
                step,
                loss: loss_value,
                mean_reward,
                std_reward,
                mean_advantage,
                total_tokens,
                gradients_applied,
                generation_time_ms: 0.0,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error: {}", e),
            )
        })??;

        Ok(metrics)
    }

    /// Unified training step with JS reward callback
    ///
    /// This method combines generation, reward scoring, and training into a single call,
    /// keeping token data in Rust memory to eliminate FFI overhead.
    ///
    /// # Arguments
    /// * `prompts` - Array of chat conversations to use as prompts
    /// * `answers` - Expected answers for each prompt (for reward functions)
    /// * `reward_fn` - JavaScript function to compute rewards: (outputs: RewardOutput[]) => Promise<number[]>
    ///
    /// # Returns
    /// * Training step result including metrics, completions, and rewards
    #[napi(
        ts_args_type = "prompts: ChatMessage[][], answers: (string | null)[], rewardFn: (err: Error | null, outputsJson: string) => Promise<number[]>"
    )]
    pub async fn train_step_auto(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        answers: Vec<Option<String>>,
        reward_fn: ThreadsafeFunction<String, Promise<Vec<f64>>>,
    ) -> Result<TrainStepResult> {
        let num_prompts = prompts.len();
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let expected_completions = num_prompts * group_size;

        let generation_start = std::time::Instant::now();

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let model_config = self.model_config.clone();
        let config = self.config.clone();

        // Build generation config
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_config.eos_token_id),
            return_logprobs: Some(true),
        };

        // === Phase 1: Generate completions (MxArray stays in Rust) ===
        let gen_result = napi::bindgen_prelude::spawn_blocking(move || {
            let mut completion_texts: Vec<String> = Vec::with_capacity(expected_completions);
            let mut prompt_texts: Vec<String> = Vec::with_capacity(num_prompts);
            let mut prompt_tokens_all: Vec<MxArray> = Vec::with_capacity(num_prompts);
            let mut completion_tokens_all: Vec<MxArray> = Vec::with_capacity(expected_completions);
            let mut completion_logprobs_all: Vec<MxArray> =
                Vec::with_capacity(expected_completions);
            let mut token_counts_all: Vec<u32> = Vec::with_capacity(expected_completions);
            let mut finish_reasons_all: Vec<String> = Vec::with_capacity(expected_completions);

            for prompt_messages in prompts.into_iter() {
                // Tokenize prompt
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(&prompt_messages, Some(true))?
                };

                // Decode prompt to string for reward function
                let prompt_array =
                    MxArray::from_uint32(&prompt_token_ids, &[prompt_token_ids.len() as i64])?;
                let prompt_text = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.decode_tokens_sync(&prompt_array)?
                };
                prompt_texts.push(prompt_text);
                prompt_tokens_all.push(prompt_array.clone());

                let prompt_2d =
                    MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;

                // Generate G completions
                for _g in 0..group_size {
                    let result = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.generate_for_training_sync(&prompt_2d, Some(gen_config.clone()))?
                    };

                    // Decode completion text
                    let text = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.decode_tokens_sync(&result.tokens)?
                    };
                    completion_texts.push(text);
                    finish_reasons_all.push(result.finish_reason.clone());

                    token_counts_all.push(result.num_tokens as u32);
                    completion_tokens_all.push(result.tokens.clone());
                    completion_logprobs_all.push(result.logprobs.clone());

                    // NOTE: Removed per-completion heavy_cleanup() for better performance
                    // MLX can batch operations more efficiently without frequent synchronization
                    // Single cleanup happens after training step completes
                }
            }

            // Light cleanup after generation phase (clears cache but doesn't stall GPU)
            synchronize_and_clear_cache();

            Ok::<_, Error>(IntermediateGenerationResult {
                completion_texts,
                prompt_texts,
                prompt_tokens: prompt_tokens_all,
                completion_tokens: completion_tokens_all,
                completion_logprobs: completion_logprobs_all,
                token_counts: token_counts_all,
                finish_reasons: finish_reasons_all,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error in generation: {}", e),
            )
        })??;

        let generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;

        // === MEMORY OPTIMIZATION: Destructure gen_result to enable early cleanup ===
        // Extract all fields so we can drop intermediate data as soon as it's no longer needed
        let IntermediateGenerationResult {
            completion_texts,    // Needed for Phase 2 (rewards) and return value
            prompt_texts,        // Only needed for Phase 2 (rewards)
            prompt_tokens,       // Needed for Phase 3 (training)
            completion_tokens,   // Needed for Phase 3 (training)
            completion_logprobs, // Needed for Phase 3 (training)
            token_counts,        // Needed for Phase 3 (training)
            finish_reasons,      // Needed for Phase 2 (rewards) and filtering
        } = gen_result;

        // Clone finish_reasons before moving to build_reward_outputs - needed for filtering
        let finish_reasons_for_filter = finish_reasons.clone();

        // === Phase 2: Build RewardOutput[] and call JS reward function ===
        let reward_outputs = build_reward_outputs(
            prompt_texts,             // Move instead of clone - prompt_texts not needed after this
            completion_texts.clone(), // Clone - we need this for return value
            answers,
            token_counts.clone(),
            finish_reasons, // Move - not needed after this (we have finish_reasons_for_filter)
            group_size as u32,
        );
        // prompt_texts is now moved/dropped, freeing memory

        // Serialize to JSON for ThreadsafeFunction (complex types don't convert directly)
        let reward_outputs_json = serde_json::to_string(&reward_outputs).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to serialize reward outputs: {}", e),
            )
        })?;

        // Call JS reward function via ThreadsafeFunction
        // The callback returns a Promise<number[]>, so we need two awaits:
        // 1. First await gets the Promise from the callback
        // 2. Second await resolves the Promise to get the actual rewards
        let promise: Promise<Vec<f64>> = reward_fn
            .call_async(Ok(reward_outputs_json))
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Reward callback call failed: {}", e),
                )
            })?;

        let rewards: Vec<f64> = promise.await.map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Reward Promise resolution failed: {}", e),
            )
        })?;

        // Validate rewards length
        if rewards.len() != expected_completions {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Expected {} rewards, got {}",
                    expected_completions,
                    rewards.len()
                ),
            ));
        }

        // === DEGENERATE OUTPUT FILTERING ===
        // Skip completions that hit the max token limit (finish_reason == "length")
        // These are likely degenerate outputs that would cause OOM in autograd
        let max_tokens_threshold = (self.config.max_new_tokens.unwrap_or(4096) as f64 * 0.9) as u32;
        let valid_indices: Vec<usize> = finish_reasons_for_filter
            .iter()
            .enumerate()
            .filter(|(i, reason)| {
                // Keep completion if it didn't hit the token limit
                // OR if it's short enough that it's not a degenerate output
                *reason != "length" || token_counts[*i] < max_tokens_threshold
            })
            .map(|(i, _)| i)
            .collect();

        let num_filtered = expected_completions - valid_indices.len();
        if num_filtered > 0 {
            info!(
                "Filtered {} degenerate completions (finish_reason='length', tokens >= {})",
                num_filtered, max_tokens_threshold
            );
        }

        // If ALL completions were filtered, skip training step entirely
        if valid_indices.is_empty() {
            warn!(
                "All {} completions hit token limit - skipping training step to prevent OOM",
                expected_completions
            );

            // Update state and return skip metrics
            let mut state = self.state.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            state.step += 1;
            let current_step = state.step;
            drop(state);

            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: u32 = token_counts.iter().sum();

            // Heavy cleanup before returning
            heavy_cleanup();

            return Ok(TrainStepResult {
                metrics: EngineStepMetrics {
                    step: current_step,
                    loss: 0.0,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: 0.0,
                },
                completions: completion_texts,
                rewards,
            });
        }

        // Calculate effective group size for training BEFORE filtering
        // This ensures filtered_count is exactly divisible by effective_group_size
        let filtered_count = valid_indices.len();
        let effective_group_size = if num_prompts > 0 {
            filtered_count / num_prompts
        } else {
            group_size
        };

        // If we don't have at least one completion per prompt, skip training
        if effective_group_size < 1 {
            warn!(
                "Only {} valid completions for {} prompts - skipping training (need at least 1 per prompt)",
                filtered_count, num_prompts
            );

            let mut state = self.state.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            state.step += 1;
            let current_step = state.step;
            drop(state);

            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: u32 = token_counts.iter().sum();

            heavy_cleanup();

            return Ok(TrainStepResult {
                metrics: EngineStepMetrics {
                    step: current_step,
                    loss: 0.0,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: 0.0,
                },
                completions: completion_texts,
                rewards,
            });
        }

        // CRITICAL: Truncate valid_indices to ensure exact divisibility
        // This guarantees: usable_count = num_prompts * effective_group_size
        // which is required by compute_advantages and prompt expansion in autograd
        let usable_count = num_prompts * effective_group_size;
        let valid_indices: Vec<usize> = if usable_count < filtered_count {
            info!(
                "Truncating {} filtered completions to {} for even group alignment ({} prompts × {} per group)",
                filtered_count, usable_count, num_prompts, effective_group_size
            );
            valid_indices.into_iter().take(usable_count).collect()
        } else {
            valid_indices
        };

        // Filter data for training - only use valid completions
        let filtered_completion_tokens: Vec<MxArray> = valid_indices
            .iter()
            .map(|&i| completion_tokens[i].clone())
            .collect();
        let filtered_completion_logprobs: Vec<MxArray> = valid_indices
            .iter()
            .map(|&i| completion_logprobs[i].clone())
            .collect();
        let filtered_token_counts: Vec<u32> =
            valid_indices.iter().map(|&i| token_counts[i]).collect();
        let filtered_rewards: Vec<f64> = valid_indices.iter().map(|&i| rewards[i]).collect();

        // Release any intermediate tensors before training phase
        synchronize_and_clear_cache();

        // === Phase 3: Train using rewards and in-memory MxArray data ===
        let training_start = std::time::Instant::now();

        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let model_config = self.model_config.clone();
        let config = self.config.clone();
        let rewards_clone = filtered_rewards.clone();
        // Use effective group size after filtering (cast to i32 for autograd API)
        let group_size_for_training = effective_group_size as i32;

        // Move only the needed MxArray data into the closure
        // completion_texts stays in outer scope for return value
        let metrics = napi::bindgen_prelude::spawn_blocking(move || {
            // Build loss config
            let loss_config = GRPOLossConfig {
                epsilon_low: config.clip_epsilon.unwrap_or(0.2),
                epsilon_high: None,
                beta: config.kl_coef.unwrap_or(0.0),
                loss_type: config
                    .loss_type
                    .clone()
                    .unwrap_or_else(|| "grpo".to_string()),
                importance_sampling_level: "token".to_string(),
                max_completion_length: config
                    .max_completion_length_for_training
                    .or(config.max_new_tokens)
                    .map(|n| n as i64),
                num_items_in_batch: Some(usable_count as f64),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
            let completion_refs: Vec<&MxArray> = filtered_completion_tokens.iter().collect();
            let logprob_refs: Vec<&MxArray> = filtered_completion_logprobs.iter().collect();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_config,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards_clone,
                group_size_for_training,
                loss_config,
            )?;

            // Check for NaN loss
            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards_clone);
                let total_tokens: u32 = filtered_token_counts.iter().sum();

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.step += 1;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            // Validate gradients - use sum-based NaN check to avoid expensive to_float32
            for (name, grad) in gradients.iter() {
                grad.eval();
                // Sum-based NaN check: if any element is NaN/Inf, the sum will be NaN/Inf
                let sum = grad.sum(None, None)?;
                sum.eval();
                let sum_val = sum.item_at_float32(0)?;
                let has_invalid = sum_val.is_nan() || sum_val.is_infinite();
                if has_invalid {
                    warn!(
                        "Gradient '{}' contains NaN/Inf values (sum={}) - SKIPPING STEP",
                        name, sum_val
                    );

                    let mut state = state_arc.write().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                    })?;
                    state.nan_gradient_count += 1;
                    state.consecutive_nan_count += 1;
                    let max_nan = config.max_nan_gradients.unwrap_or(100) as u64;

                    if state.nan_gradient_count >= max_nan {
                        return Err(Error::new(
                            Status::GenericFailure,
                            format!(
                                "Training stopped: exceeded maximum NaN gradient count ({}/{})",
                                state.nan_gradient_count, max_nan
                            ),
                        ));
                    }

                    let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold {
                        state.needs_emergency_save = true;
                    }

                    state.step += 1;
                    let current_step = state.step;
                    drop(state);

                    let (mean_reward, std_reward) = compute_reward_stats(&rewards_clone);
                    let total_tokens: u32 = filtered_token_counts.iter().sum();
                    synchronize_and_clear_cache();

                    return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                        step: current_step,
                        loss: loss_value,
                        mean_reward,
                        std_reward,
                        mean_advantage: 0.0,
                        total_tokens: total_tokens as i32,
                        gradients_applied: false,
                        generation_time_ms,
                        training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                    });
                }
            }

            // Fused value and norm clipping
            let grad_clip_value = config.gradient_clip_value.unwrap_or(1.0);
            let grad_refs: HashMap<String, &MxArray> =
                gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
            let gradients = GradientUtils::clip_grad_value_and_norm(
                grad_refs,
                grad_clip_value,
                config.gradient_clip_norm,
            )?;

            // Reset consecutive NaN count
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            state.consecutive_nan_count = 0;

            accumulate_gradients(&mut state, gradients)?;
            state.micro_step += 1;

            let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
            let gradients_applied = if state.micro_step >= grad_acc_steps {
                let grads = state.accumulated_gradients.take().ok_or_else(|| {
                    Error::new(Status::GenericFailure, "No accumulated gradients")
                })?;

                let lr = config.learning_rate.unwrap_or(1e-6) / grad_acc_steps as f64;
                drop(state);

                let mut model_mut = model_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model write lock")
                })?;

                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                model_mut.apply_gradients(grads_refs, lr)?;
                drop(model_mut);

                // Release accumulated gradients
                drop(grads);

                debug!("Applied gradients with lr: {}", lr);

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.accumulated_gradients = None;
                state.micro_step = 0;
                state.step += 1;
                state.epoch_steps += 1;
                drop(state);

                // CRITICAL: Release gradient tensors and computation graphs
                heavy_cleanup();

                true
            } else {
                state.step += 1;
                state.epoch_steps += 1;
                drop(state);

                // Cleanup even when not applying (release autograd graph)
                heavy_cleanup();

                false
            };

            // Compute metrics
            let (mean_reward, std_reward) = compute_reward_stats(&rewards_clone);
            let total_tokens: u32 = filtered_token_counts.iter().sum();

            // Update epoch accumulators
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.epoch_loss_sum += loss_value;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }

            // Compute mean advantage
            let rewards_f32: Vec<f32> = rewards_clone.iter().map(|&r| r as f32).collect();
            let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards_clone.len() as i64])?;
            let advantages =
                compute_advantages(&rewards_array, group_size_for_training, "group".to_string())?;
            let adv_data = advantages.to_float32()?;
            let mean_advantage =
                adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len() as f64;

            synchronize_and_clear_cache();

            // Periodic heavy cleanup to prevent GPU timeout in long-running training
            let heavy_cleanup_interval = config.heavy_cleanup_interval.unwrap_or(25);
            let memory_cleanup_threshold = config.memory_cleanup_threshold.unwrap_or(0.0);
            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                maybe_heavy_cleanup(&mut state, heavy_cleanup_interval, memory_cleanup_threshold);
            }

            let step = state_arc
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?
                .step;

            Ok(EngineStepMetrics {
                step,
                loss: loss_value,
                mean_reward,
                std_reward,
                mean_advantage,
                total_tokens: total_tokens as i32,
                gradients_applied,
                generation_time_ms,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error in training: {}", e),
            )
        })??;

        Ok(TrainStepResult {
            metrics,
            completions: completion_texts,
            rewards,
        })
    }

    /// Score completions using registered built-in rewards
    ///
    /// # Arguments
    /// * `prompts` - Prompt texts (expanded to match completions)
    /// * `completions` - Completion texts to score
    #[napi]
    pub fn score_completions(&self, prompts: Vec<String>, completions: Vec<String>) -> Vec<f64> {
        if self.reward_registry.is_empty() {
            return vec![0.0; completions.len()];
        }

        self.reward_registry.score_batch(&prompts, &completions)
    }

    /// Get current training step
    #[napi(getter)]
    pub fn step(&self) -> Result<i64> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        Ok(state.step)
    }

    /// Get current epoch
    #[napi(getter)]
    pub fn epoch(&self) -> Result<i32> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        Ok(state.epoch)
    }

    /// Start a new epoch
    #[napi]
    pub fn start_epoch(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;

        state.epoch += 1;
        state.epoch_loss_sum = 0.0;
        state.epoch_reward_sum = 0.0;
        state.epoch_steps = 0;
        state.epoch_tokens = 0;

        info!("Starting epoch {}", state.epoch);
        Ok(())
    }

    /// End the current epoch and get metrics
    #[napi]
    pub fn end_epoch(&self, epoch_time_secs: f64) -> Result<EngineEpochMetrics> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;

        let avg_loss = if state.epoch_steps > 0 {
            state.epoch_loss_sum / state.epoch_steps as f64
        } else {
            0.0
        };

        let avg_reward = if state.epoch_steps > 0 {
            state.epoch_reward_sum / state.epoch_steps as f64
        } else {
            0.0
        };

        info!(
            "Epoch {} complete: avg_loss={:.6}, avg_reward={:.4}, steps={}",
            state.epoch, avg_loss, avg_reward, state.epoch_steps
        );

        Ok(EngineEpochMetrics {
            epoch: state.epoch,
            avg_loss,
            avg_reward,
            total_steps: state.epoch_steps,
            total_tokens: state.epoch_tokens,
            epoch_time_secs,
        })
    }

    /// Reset the engine for a fresh training run
    #[napi]
    pub fn reset(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;

        *state = EngineState::default();
        synchronize_and_clear_cache();

        info!("Training engine reset");
        Ok(())
    }

    /// Check if reward registry has any rewards registered
    #[napi(getter)]
    pub fn has_builtin_rewards(&self) -> bool {
        !self.reward_registry.is_empty()
    }

    /// Get names of registered reward functions
    #[napi(getter)]
    pub fn reward_names(&self) -> Vec<String> {
        self.reward_registry.names()
    }

    /// Get current micro-step within gradient accumulation
    #[napi(getter)]
    pub fn micro_step(&self) -> Result<i32> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        Ok(state.micro_step)
    }

    /// Check if an emergency checkpoint should be saved
    /// This flag is set when consecutive NaN gradients reach the threshold
    #[napi(getter)]
    pub fn needs_emergency_save(&self) -> Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        Ok(state.needs_emergency_save)
    }

    /// Get current NaN gradient count
    #[napi(getter)]
    pub fn nan_gradient_count(&self) -> Result<i64> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        Ok(state.nan_gradient_count as i64)
    }

    /// Clear the emergency save flag (call after saving emergency checkpoint)
    #[napi]
    pub fn clear_emergency_save_flag(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
        state.needs_emergency_save = false;
        Ok(())
    }
}

// =============================================================================
// Helper types
// =============================================================================

/// Internal result from generation phase (not exposed to NAPI)
/// Keeps MxArray data in Rust memory for efficient training
struct IntermediateGenerationResult {
    /// Generated completion texts (for reward function and return value)
    completion_texts: Vec<String>,
    /// Formatted prompt texts (for reward function)
    prompt_texts: Vec<String>,
    /// Prompt tokens as MxArray (for training)
    prompt_tokens: Vec<MxArray>,
    /// Completion tokens as MxArray (for training)
    completion_tokens: Vec<MxArray>,
    /// Completion log probabilities as MxArray (for training)
    completion_logprobs: Vec<MxArray>,
    /// Token counts for each completion
    token_counts: Vec<u32>,
    /// Finish reasons for each completion ("eos", "length", or "repetition")
    finish_reasons: Vec<String>,
}

// =============================================================================
// Helper functions
// =============================================================================

/// Accumulate gradients into state
/// Gradients are already validated before calling this function.
/// We eval() each accumulated gradient to materialize it and allow MLX to free the computation graph.
fn accumulate_gradients(
    state: &mut EngineState,
    new_grads: HashMap<String, MxArray>,
) -> Result<()> {
    // Skip redundant validation - already done in train_step_auto before calling this
    // The expensive to_float32() validation was causing ~2GB memory overhead per step

    match &mut state.accumulated_gradients {
        Some(acc) => {
            for (name, grad) in new_grads {
                if let Some(existing) = acc.get_mut(&name) {
                    let summed = existing.add(&grad)?;
                    // CRITICAL: eval() to materialize the result, allowing MLX to free
                    // the computation graph from the add operation
                    summed.eval();
                    *existing = summed;
                } else {
                    // First accumulation - just eval and store
                    grad.eval();
                    acc.insert(name, grad);
                }
            }
        }
        None => {
            // First step - eval all gradients to materialize them
            let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
            for (name, grad) in new_grads {
                grad.eval();
                evaluated_grads.insert(name, grad);
            }
            state.accumulated_gradients = Some(evaluated_grads);
        }
    }
    Ok(())
}

/// Perform heavy cleanup if interval has been reached or memory threshold exceeded
/// Returns true if cleanup was performed
fn maybe_heavy_cleanup(
    state: &mut EngineState,
    heavy_cleanup_interval: i32,
    memory_threshold: f64,
) -> bool {
    let steps_since_cleanup = state.step - state.last_heavy_cleanup_step;

    // Step-based trigger: cleanup every N steps
    let step_triggered = steps_since_cleanup >= heavy_cleanup_interval as i64;

    // Memory-based trigger: only if a positive threshold is set
    // (passing 0.0 disables memory-based triggering)
    let memory_triggered = if memory_threshold > 0.0 {
        let active_mem = get_active_memory();
        let cache_mem = get_cache_memory();
        let total_mem = active_mem + cache_mem;
        total_mem > memory_threshold
    } else {
        false
    };

    if step_triggered || memory_triggered {
        heavy_cleanup();
        state.last_heavy_cleanup_step = state.step;
        return true;
    }
    false
}

/// Compute reward statistics
fn compute_reward_stats(rewards: &[f64]) -> (f64, f64) {
    if rewards.is_empty() {
        return (0.0, 0.0);
    }

    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    let std = variance.sqrt();

    (mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_reward_stats() {
        use std::f64::consts::SQRT_2;
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = compute_reward_stats(&rewards);
        assert!((mean - 3.0).abs() < 0.001);
        assert!((std - SQRT_2).abs() < 0.01);
    }

    #[test]
    fn test_compute_reward_stats_empty() {
        let rewards: Vec<f64> = vec![];
        let (mean, std) = compute_reward_stats(&rewards);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }
}
