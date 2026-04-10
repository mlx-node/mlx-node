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
/// const model = await Qwen3Model.load(modelPath);
/// const engine = new GRPOTrainingEngine(model, config);
/// engine.registerBuiltinReward({ rewardType: 'ToolUse', allowedTools: ['search'] });
///
/// for (const batch of dataset) {
///   const metrics = await engine.trainStep(batch.prompts);
///   console.log(metrics);
/// }
/// ```
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use tracing::{info, warn};

use crate::grpo::loss::GRPOLossConfig;
use crate::grpo::rewards::{
    BuiltinRewardConfig, JsonSchemaReward, LengthReward, RewardRegistry, ToolUseReward,
    XMLFormatReward,
};
use crate::models::qwen3::{GenerationConfig, Qwen3Cmd, Qwen3Model};
use crate::models::qwen3_5::model::{Qwen3_5Model, Qwen35Cmd};
use crate::models::qwen3_5_moe::model::{Qwen3_5MoeModel, Qwen35MoeCmd};
use crate::tokenizer::{ChatMessage, ToolDefinition};
use crate::tools::build_reward_outputs;
use crate::training_model::{
    GenerationPlainData, ModelType, TrainStepPlainMetrics, TrainingDispatch,
};

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
    /// Maximum completion length for both generation and training (default: 256)
    /// Matches Python TRL's max_completion_length config.
    pub max_completion_length: Option<i32>,
    /// Sampling temperature (default: 0.8)
    pub temperature: Option<f64>,
    /// Top-p (nucleus) sampling (default: 0.95)
    pub top_p: Option<f64>,
    /// Top-k sampling (optional)
    pub top_k: Option<i32>,
    /// Repetition penalty (default: 1.1)
    pub repetition_penalty: Option<f64>,
    /// Presence penalty (0.0 = disabled). Subtracts a flat penalty from logits of any
    /// token that appeared at least once in context.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty (0.0 = disabled). Subtracts penalty * occurrence_count from
    /// logits of each token in context.
    pub frequency_penalty: Option<f64>,

    // === NaN gradient protection ===
    /// Maximum allowed NaN gradient occurrences before stopping training (default: 100)
    /// When exceeded, training will stop with an error to prevent model corruption.
    pub max_nan_gradients: Option<i64>,
    /// Consecutive NaN gradients that trigger emergency checkpoint (default: 5)
    /// When reached, the needs_emergency_save flag is set for the TypeScript layer.
    pub emergency_save_threshold: Option<i32>,
    /// Enable detailed NaN/Inf detection with per-element counts (default: false)
    /// When false (default), uses GPU-native has_nan_or_inf() which only transfers a single
    /// boolean to CPU. When true, transfers the entire gradient tensor to CPU for detailed
    /// per-element analysis - useful for debugging but has significant performance overhead
    /// for large models (e.g., 2.4GB for Qwen3-0.6B).
    pub verbose_nan_detection: Option<bool>,

    // === Chat template parameters ===
    /// Enable thinking mode for Qwen3 models (default: true)
    /// When false, adds empty <think></think> tags to disable model thinking.
    /// This is useful for tool-use training where you want direct outputs.
    pub enable_thinking: Option<bool>,

    // === Tool calling ===
    /// Tool definitions for function calling
    /// When provided, tools are included in the chat template so the model
    /// can generate tool calls. This is essential for tool-use training.
    pub tools: Option<Vec<ToolDefinition>>,

    // === Memory optimization ===
    /// Batch chunk size for LM head computation (memory optimization).
    /// When set, the LM head (hidden_states -> logits) is computed in chunks
    /// of this size to reduce peak memory usage.
    /// Default: None (no chunking, full batch at once)
    /// Recommended: 2 for batch_size >= 4 with large vocabularies (e.g., 151936)
    /// This reduces peak memory from ~1.2GB to ~300MB for Qwen3 (vocab=151936).
    pub lm_head_chunk_size: Option<i32>,

    /// Batch chunk size for transformer forward pass (memory optimization).
    /// When set, the transformer layers process the batch in chunks of this size,
    /// reducing peak memory from O(batch × heads × seq²) for attention.
    /// Default: None (no chunking, full batch at once)
    /// Recommended: 4 for batch_size >= 4 with groupSize >= 4
    /// Memory savings: ~70-80% for batch=4, groupSize=4 (16 sequences → 4 at a time)
    pub forward_chunk_size: Option<i32>,

    /// Chunk size for vocabulary dimension in cross-entropy computation.
    /// When computing logsumexp over large vocabularies (e.g., Qwen3's 151,936 tokens),
    /// the computation is split into chunks of this size to reduce peak memory usage.
    /// Default: 65536 (2^16)
    /// Recommended: 65536 for Qwen3 (vocab=151936) splits into 3 chunks
    /// Set to a larger value to reduce chunking overhead or smaller for tighter memory constraints.
    pub vocab_chunk_size: Option<i32>,

    // === Parallel batch generation ===
    /// Enable true parallel batch generation (default: false).
    /// When true, all N*G sequences are processed in parallel using batched FFI
    /// with per-sequence RoPE offsets. This provides 2-4x speedup for GRPO training.
    /// When false, uses the sequential generation (process one prompt at a time,
    /// then expand KV cache for G completions).
    pub use_parallel_batch_generation: Option<bool>,

    /// Enable gradient checkpointing (default: true).
    /// When true, each transformer layer's activations are discarded during the forward
    /// pass and recomputed during backward, reducing peak memory from O(num_layers) to O(1)
    /// for intermediate states. For Qwen3.5 0.8B, this reduces autograd peak from ~105GB to ~11GB.
    /// The trade-off is ~30% more compute (one extra forward pass per layer during backward).
    pub gradient_checkpointing: Option<bool>,

    /// Optimizer type: "sgd" or "adamw" (default: "adamw")
    pub optimizer_type: Option<String>,
    /// AdamW beta1 (default: 0.9)
    pub adamw_beta1: Option<f64>,
    /// AdamW beta2 (default: 0.999)
    pub adamw_beta2: Option<f64>,
    /// AdamW epsilon (default: 1e-8)
    pub adamw_eps: Option<f64>,
    /// Weight decay for AdamW (default: 0.01)
    pub weight_decay: Option<f64>,
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
            max_completion_length: Some(256),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: None,
            repetition_penalty: Some(1.1),
            presence_penalty: None,
            frequency_penalty: None,
            max_nan_gradients: Some(100),
            emergency_save_threshold: Some(5),
            verbose_nan_detection: Some(false),
            enable_thinking: Some(true),
            tools: None,
            lm_head_chunk_size: Some(2), // Default: 2 (chunked for memory efficiency)
            forward_chunk_size: None,    // Default: no chunking
            vocab_chunk_size: Some(65536), // Default: 2^16 chunks for large vocabularies
            use_parallel_batch_generation: Some(false), // Default: use sequential for stability
            gradient_checkpointing: Some(true), // Default: enable for memory efficiency
            optimizer_type: Some("adamw".to_string()),
            adamw_beta1: Some(0.9),
            adamw_beta2: Some(0.999),
            adamw_eps: Some(1e-8),
            weight_decay: Some(0.01),
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
    /// Standard deviation of advantages
    pub std_advantage: f64,
    /// Total tokens generated this step
    pub total_tokens: i32,
    /// Whether gradients were applied
    pub gradients_applied: bool,
    /// Time for generation (ms)
    pub generation_time_ms: f64,
    /// Time for training (ms)
    pub training_time_ms: f64,
    /// Peak memory usage this step (MB)
    pub peak_memory_mb: f64,
    /// Active memory at end of step (MB)
    pub active_memory_mb: f64,
}

/// Convert from NAPI EngineStepMetrics to mlx-db EngineStepMetrics
impl From<&EngineStepMetrics> for mlx_db::EngineStepMetrics {
    fn from(m: &EngineStepMetrics) -> Self {
        mlx_db::EngineStepMetrics {
            step: m.step,
            loss: m.loss,
            mean_reward: m.mean_reward,
            std_reward: m.std_reward,
            mean_advantage: m.mean_advantage,
            std_advantage: m.std_advantage,
            total_tokens: m.total_tokens,
            gradients_applied: m.gradients_applied,
            generation_time_ms: m.generation_time_ms,
            training_time_ms: m.training_time_ms,
            peak_memory_mb: m.peak_memory_mb,
            active_memory_mb: m.active_memory_mb,
        }
    }
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
    /// Finish reasons for each completion ("stop", "length", or "repetition")
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

/// Result from train_step_auto_with_recording including optional full RewardOutput data
#[napi(object)]
#[derive(Clone)]
pub struct TrainStepResultWithOutputs {
    /// Training metrics
    pub metrics: EngineStepMetrics,
    /// Generated completion texts (for TUI logging)
    pub completions: Vec<String>,
    /// Computed reward values (for TUI logging)
    pub rewards: Vec<f64>,
    /// Full RewardOutput data as JSON (only populated when record_outputs is true)
    /// This enables zero-copy persistence of training outputs
    pub outputs_json: Option<String>,
    /// Actual token counts for each completion (for accurate TUI display)
    pub completion_lengths: Vec<i32>,
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
    epoch_reward_sum: f64,
    epoch_steps: i64,
    epoch_tokens: i64,
    /// Cumulative NaN gradient count (mirrored from model thread metrics)
    nan_gradient_count: u64,
    /// Consecutive NaN gradient count (for emergency checkpoint detection)
    consecutive_nan_count: u32,
    /// Flag indicating an emergency checkpoint should be saved
    needs_emergency_save: bool,
    /// Lifecycle generation — bumped by `reset()` so an in-flight
    /// `train_step*()` can detect that its stale post-await writeback would
    /// resurrect cleared state and skip the update.
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
            epoch_reward_sum: 0.0,
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

/// GRPO Training Engine
///
/// Thin coordinator that routes all MLX operations through the model thread.
/// No MxArrays or model state live here — only plain data crosses the boundary.
#[napi]
pub struct GRPOTrainingEngine {
    /// Dispatch handle for sending commands to the model thread
    dispatch: TrainingDispatch,
    /// Model type (carries config for identifying model family)
    model_type: ModelType,
    /// Engine configuration
    config: GRPOEngineConfig,
    /// Reward registry (built-in rewards)
    reward_registry: RewardRegistry,
    /// Training state (epoch counters and emergency save flag)
    state: Arc<RwLock<EngineState>>,
    // No optimizer — lives on model thread in ModelThreadTrainingState
}

#[napi]
impl GRPOTrainingEngine {
    /// Create a new training engine from a Qwen3 model
    ///
    /// # Arguments
    /// * `model` - The Qwen3 model (must be loaded via load())
    /// * `config` - Engine configuration
    #[napi(constructor)]
    pub fn new(model: &Qwen3Model, config: GRPOEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen3(model.get_config());

        info!(
            "Creating training engine: {} layers, {} hidden, eos_token_id={}, pad_token_id={}",
            model_type.num_layers(),
            model_type.hidden_size(),
            model_type.eos_token_id(),
            model_type.pad_token_id()
        );

        // Extract the cmd_sender from the model's thread
        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        // Send InitTraining to set up optimizer + state on model thread
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen3Cmd::InitTraining {
                config: Box::new(config.clone()),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen3(sender),
            model_type,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new training engine from a Qwen3.5 dense model
    #[napi(factory)]
    pub fn from_qwen35(model: &Qwen3_5Model, config: GRPOEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Dense(model.config.clone());

        info!(
            "Creating training engine (Qwen3.5 Dense): {} layers, {} hidden",
            model_type.num_layers(),
            model_type.hidden_size()
        );

        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen35Cmd::InitTraining {
                config: Box::new(config.clone()),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen35Dense(sender),
            model_type,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
        })
    }

    /// Create a new training engine from a Qwen3.5 MoE model
    #[napi(factory)]
    pub fn from_qwen35_moe(model: &Qwen3_5MoeModel, config: GRPOEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Moe(model.config.clone());

        info!(
            "Creating training engine (Qwen3.5 MoE): {} layers, {} hidden",
            model_type.num_layers(),
            model_type.hidden_size()
        );

        let sender = model
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Model thread not running"))?
            .clone();

        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen35MoeCmd::InitTraining {
                config: Box::new(config.clone()),
                model_type: model_type.clone(),
                reply: tx,
            })
            .map_err(|_| Error::from_reason("Model thread has exited"))?;
        rx.blocking_recv()
            .map_err(|_| Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            dispatch: TrainingDispatch::Qwen35Moe(sender),
            model_type,
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
    /// 3. Compute GRPO loss and gradients (on model thread)
    /// 4. Apply gradients (respecting accumulation steps, on model thread)
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
        let gen_config = self.build_gen_config();

        // Reject dispatch from an invalidated handle and snapshot the
        // lifecycle generation before any await points so that if `reset()`
        // runs concurrently, our stale post-await writeback is discarded
        // and does not resurrect cleared state.
        let start_generation = self.ensure_valid_snapshot_generation()?;

        // === Phase 1: Generate completions via model thread (single batched call) ===
        let gen_data = self
            .dispatch_generate(
                &prompts,
                group_size,
                gen_config.clone(),
                self.config.enable_thinking,
                self.config.tools.clone(),
            )
            .await?;

        let total_tokens: i32 = gen_data.token_counts.iter().map(|&tc| tc as i32).sum();

        let generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        let training_start = std::time::Instant::now();

        // === Phase 2: Train via model thread ===
        // Re-check invalidation after the generate await: a concurrent
        // `reset()` may have fired during Phase 1, and we must not dispatch
        // a training step against a newly re-initialized training_state on
        // behalf of an old handle.
        self.ensure_valid()?;
        let loss_config = self.build_loss_config(num_prompts, group_size);
        let metrics = self
            .dispatch_train_step(rewards.clone(), group_size as i32, loss_config, None)
            .await?;

        let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;

        // Compute reward stats (plain data, safe on NAPI thread)
        let (mean_reward, std_reward) = compute_reward_stats(&rewards);

        // Update engine state from model thread metrics — gated on generation
        // so a concurrent reset wins cleanly.
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

                state.epoch_loss_sum += metrics.loss;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }
        }

        Ok(EngineStepMetrics {
            step: metrics.step,
            loss: metrics.loss,
            mean_reward,
            std_reward,
            mean_advantage: metrics.mean_advantage,
            std_advantage: metrics.std_advantage,
            total_tokens: metrics.total_tokens,
            gradients_applied: metrics.gradients_applied,
            generation_time_ms,
            training_time_ms,
            peak_memory_mb: metrics.peak_memory_mb,
            active_memory_mb: metrics.active_memory_mb,
        })
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
        self.ensure_valid()?;
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let gen_config = self.build_gen_config();

        // Single batched dispatch — the returned GenerationPlainData already
        // holds all prompts' data in prompt-major order.
        let gen_data = self
            .dispatch_generate(
                &prompts,
                group_size,
                gen_config.clone(),
                self.config.enable_thinking,
                self.config.tools.clone(),
            )
            .await?;

        let total_completions = gen_data.completion_texts.len();
        let mut all_texts = Vec::with_capacity(total_completions);
        let mut all_tokens = Vec::new();
        let mut all_logprobs = Vec::new();
        let mut all_lengths = Vec::with_capacity(total_completions);
        let mut all_reasons = Vec::with_capacity(total_completions);

        for i in 0..total_completions {
            all_texts.push(gen_data.completion_texts[i].clone());
            all_lengths.push(gen_data.completion_tokens[i].len() as i32);
            all_tokens.extend(gen_data.completion_tokens[i].iter().map(|&t| t as i64));
            all_logprobs.extend(gen_data.completion_logprobs[i].iter().map(|&l| l as f64));
            all_reasons.push(gen_data.finish_reasons[i].clone());
        }

        Ok(GenerateBatchResult {
            completion_texts: all_texts,
            completion_tokens: all_tokens,
            completion_logprobs: all_logprobs,
            completion_lengths: all_lengths,
            finish_reasons: all_reasons,
        })
    }

    /// Run a training step with pre-generated completions
    ///
    /// Uses the cached MxArrays from the most recent generate_batch_for_training
    /// call on the model thread. The generation_result parameter is used only for
    /// validation (the actual MxArrays are cached on the model thread).
    ///
    /// # Arguments
    /// * `prompts` - Array of chat conversations to use as prompts
    /// * `rewards` - Reward values for each completion (num_prompts * group_size)
    /// * `generation_result` - Pre-generated completion data (used for validation)
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

        // Reject dispatch from an invalidated handle and snapshot the
        // lifecycle generation before the train_step await so any
        // concurrent reset wins cleanly.
        let start_generation = self.ensure_valid_snapshot_generation()?;

        // Dispatch train step to model thread (uses cached MxArrays from generate phase)
        let loss_config = self.build_loss_config(num_prompts, group_size);
        let metrics = self
            .dispatch_train_step(rewards.clone(), group_size as i32, loss_config, None)
            .await?;

        let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;

        let (mean_reward, std_reward) = compute_reward_stats(&rewards);
        let total_tokens: i32 = generation_result.completion_lengths.iter().sum();

        // Update engine state — gated on generation.
        {
            let mut state = self.state.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            if state.generation == start_generation {
                state.step = metrics.step;
                state.epoch_steps += 1;
                state.nan_gradient_count = metrics.nan_gradient_count;

                if !metrics.gradients_applied && metrics.nan_gradient_count > 0 {
                    state.consecutive_nan_count += 1;
                    let emergency_threshold =
                        self.config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold
                        && !state.needs_emergency_save
                    {
                        state.needs_emergency_save = true;
                    }
                } else {
                    state.consecutive_nan_count = 0;
                }

                state.epoch_loss_sum += metrics.loss;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }
        }

        Ok(EngineStepMetrics {
            step: metrics.step,
            loss: metrics.loss,
            mean_reward,
            std_reward,
            mean_advantage: metrics.mean_advantage,
            std_advantage: metrics.std_advantage,
            total_tokens: metrics.total_tokens,
            gradients_applied: metrics.gradients_applied,
            generation_time_ms: 0.0,
            training_time_ms,
            peak_memory_mb: metrics.peak_memory_mb,
            active_memory_mb: metrics.active_memory_mb,
        })
    }

    /// Unified training step with JS reward callback and optional output recording
    ///
    /// Generates completions via the model thread, calls the JS reward function
    /// with plain data, then dispatches the training step to the model thread.
    ///
    /// # Arguments
    /// * `prompts` - Array of chat conversations to use as prompts
    /// * `reward_fn` - JavaScript function to compute rewards
    /// * `record_outputs` - If true, return the serialized RewardOutput JSON
    ///
    /// # Returns
    /// * Training step result including metrics, completions, rewards, and optionally outputs_json
    #[napi(
        ts_args_type = "prompts: ChatMessage[][], rewardFn: (err: Error | null, outputsJson: string) => Promise<number[]>, recordOutputs: boolean"
    )]
    pub async fn train_step_auto(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        reward_fn: ThreadsafeFunction<String, Promise<Vec<f64>>>,
        record_outputs: bool,
    ) -> Result<TrainStepResultWithOutputs> {
        let num_prompts = prompts.len();
        let group_size = self.config.group_size.unwrap_or(4) as usize;
        let expected_completions = num_prompts * group_size;

        let generation_start = std::time::Instant::now();
        let gen_config = self.build_gen_config();

        // Reject dispatch from an invalidated handle and snapshot the
        // lifecycle generation before any await points so that if `reset()`
        // runs concurrently, our stale post-await writebacks are discarded
        // and do not resurrect cleared state.
        let start_generation = self.ensure_valid_snapshot_generation()?;

        // === Phase 1: Generate completions via model thread ===
        info!(
            "Phase 1: Generating {} completions ({} prompts × {} groups)",
            expected_completions, num_prompts, group_size
        );

        let mut all_completion_texts: Vec<String> = Vec::with_capacity(expected_completions);
        let mut all_prompt_texts: Vec<String> = Vec::with_capacity(num_prompts);
        let mut all_token_counts: Vec<u32> = Vec::with_capacity(expected_completions);
        let mut all_finish_reasons: Vec<String> = Vec::with_capacity(expected_completions);

        // Single batched dispatch — accumulates completions for all prompts in
        // prompt-major order on the model thread.
        let gen_data = self
            .dispatch_generate(
                &prompts,
                group_size,
                gen_config.clone(),
                self.config.enable_thinking,
                self.config.tools.clone(),
            )
            .await?;

        // Collect one prompt text per prompt (prompt_texts is repeated per
        // completion in prompt-major order, so stride by group_size).
        for p in 0..num_prompts {
            let base = p * group_size;
            if let Some(pt) = gen_data.prompt_texts.get(base) {
                all_prompt_texts.push(pt.clone());
            }
        }

        for i in 0..gen_data.completion_texts.len() {
            all_completion_texts.push(gen_data.completion_texts[i].clone());
            all_token_counts.push(gen_data.token_counts[i]);
            all_finish_reasons.push(gen_data.finish_reasons[i].clone());
        }

        let generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Phase 1 complete: generated in {:.1}s",
            generation_time_ms / 1000.0
        );

        // === Phase 2: Build RewardOutput[] and call JS reward function ===
        info!("Phase 2: Computing rewards via JS callback...");
        let reward_outputs = build_reward_outputs(
            all_prompt_texts,
            all_completion_texts.clone(),
            all_token_counts.clone(),
            all_finish_reasons.clone(),
            group_size as u32,
        );

        // Serialize to JSON - always needed for reward callback
        let reward_outputs_json = serde_json::to_string(&reward_outputs).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to serialize reward outputs: {}", e),
            )
        })?;

        // Keep a copy for return value if recording is enabled
        let outputs_json_for_return = if record_outputs {
            Some(reward_outputs_json.clone())
        } else {
            None
        };

        // Call JS reward function
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

        // === DEGENERATE OUTPUT FILTERING (per-prompt balanced) ===
        //
        // The autograd / advantages paths require a rectangular layout:
        // `valid_indices.len() == num_prompts * effective_group_size`.
        // We therefore filter per prompt and then truncate every prompt to the
        // minimum survivor count, so uneven drops across prompts can't break
        // divisibility downstream.
        let max_tokens_threshold =
            (self.config.max_completion_length.unwrap_or(4096) as f64 * 0.9) as u32;
        let (valid_indices, balanced_effective_group_size, per_prompt_survivor_counts) =
            compute_balanced_valid_indices(
                &all_finish_reasons,
                &all_token_counts,
                num_prompts,
                group_size,
                max_tokens_threshold,
            );

        let num_filtered = expected_completions - valid_indices.len();
        if num_filtered > 0 {
            let min_survivors = per_prompt_survivor_counts
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            let max_survivors = per_prompt_survivor_counts
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            info!(
                "Filtered {} degenerate completions (finish_reason='length', tokens >= {}); \
                 per-prompt survivors: min={} max={} counts={:?}, effective_group_size={}",
                num_filtered,
                max_tokens_threshold,
                min_survivors,
                max_survivors,
                per_prompt_survivor_counts,
                balanced_effective_group_size
            );
        }

        if valid_indices.is_empty() {
            warn!(
                "All {} completions hit token limit - skipping training step to prevent OOM",
                expected_completions
            );

            // Re-check invalidation after the JS reward callback await —
            // a concurrent `reset()` may have fired while JS was running.
            self.ensure_valid()?;
            // Bump ts.step on the model thread (it's the single source of truth)
            // and drop stale MxArrays in the same round-trip. Mirror the result
            // into the engine's read-through cache — gated on generation.
            let current_step = self.dispatch_bump_skipped_step().await?;

            {
                let mut state = self.state.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                if state.generation == start_generation {
                    state.step = current_step;
                }
            }

            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: u32 = all_token_counts.iter().sum();

            return Ok(TrainStepResultWithOutputs {
                metrics: EngineStepMetrics {
                    step: current_step,
                    loss: 0.0,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: 0.0,
                    peak_memory_mb: 0.0,
                    active_memory_mb: 0.0,
                },
                completions: all_completion_texts,
                rewards,
                outputs_json: outputs_json_for_return,
                completion_lengths: all_token_counts.iter().map(|&x| x as i32).collect(),
            });
        }

        let filtered_count = valid_indices.len();
        // By construction from compute_balanced_valid_indices:
        // filtered_count == num_prompts * balanced_effective_group_size
        let effective_group_size = balanced_effective_group_size;
        debug_assert_eq!(
            filtered_count,
            num_prompts * effective_group_size,
            "balanced filter must produce a rectangular layout"
        );

        if effective_group_size < 1 {
            warn!(
                "Only {} valid completions for {} prompts - skipping training",
                filtered_count, num_prompts
            );

            // Re-check invalidation after the JS reward callback await —
            // a concurrent `reset()` may have fired while JS was running.
            self.ensure_valid()?;
            // Bump ts.step on the model thread (it's the single source of truth)
            // and drop stale MxArrays in the same round-trip. Mirror the result
            // into the engine's read-through cache — gated on generation.
            let current_step = self.dispatch_bump_skipped_step().await?;

            {
                let mut state = self.state.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                if state.generation == start_generation {
                    state.step = current_step;
                }
            }

            let (mean_reward, std_reward) = compute_reward_stats(&rewards);
            let total_tokens: u32 = all_token_counts.iter().sum();

            return Ok(TrainStepResultWithOutputs {
                metrics: EngineStepMetrics {
                    step: current_step,
                    loss: 0.0,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: 0.0,
                    peak_memory_mb: 0.0,
                    active_memory_mb: 0.0,
                },
                completions: all_completion_texts,
                rewards,
                outputs_json: outputs_json_for_return,
                completion_lengths: all_token_counts.iter().map(|&x| x as i32).collect(),
            });
        }

        // Filter rewards for the valid completions
        let filtered_rewards: Vec<f64> = valid_indices.iter().map(|&i| rewards[i]).collect();

        // === Phase 3: Train via model thread ===
        info!(
            "Phase 3: Training with {} valid completions (filtered {})",
            valid_indices.len(),
            num_filtered
        );
        let training_start = std::time::Instant::now();

        let usable_count = num_prompts * effective_group_size;
        let loss_config = GRPOLossConfig {
            epsilon_low: self.config.clip_epsilon.unwrap_or(0.2),
            epsilon_high: None,
            beta: self.config.kl_coef.unwrap_or(0.0),
            loss_type: self
                .config
                .loss_type
                .clone()
                .unwrap_or_else(|| "grpo".to_string()),
            importance_sampling_level: "token".to_string(),
            max_completion_length: self.config.max_completion_length.map(|n| n as i64),
            num_items_in_batch: Some(usable_count as f64),
            gradient_accumulation_steps: self.config.gradient_accumulation_steps.unwrap_or(1)
                as i64,
            lm_head_chunk_size: self.config.lm_head_chunk_size.map(|n| n as i64),
            forward_chunk_size: self.config.forward_chunk_size.map(|n| n as i64),
            vocab_chunk_size: self.config.vocab_chunk_size.map(|n| n as i64),
        };

        // Final re-check before the training dispatch: this is the last
        // await gap where a concurrent `reset()` could have invalidated
        // this handle between the reward callback and the training step.
        self.ensure_valid()?;
        let metrics = self
            .dispatch_train_step(
                filtered_rewards.clone(),
                effective_group_size as i32,
                loss_config,
                Some(valid_indices.clone()),
            )
            .await?;

        let training_time_ms = training_start.elapsed().as_secs_f64() * 1000.0;

        let (mean_reward, std_reward) = compute_reward_stats(&filtered_rewards);
        let total_tokens: u32 = all_token_counts.iter().sum();

        // Update engine state — gated on generation so a concurrent
        // reset wins cleanly.
        {
            let mut state = self.state.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;
            if state.generation == start_generation {
                state.step = metrics.step;
                state.epoch_steps += 1;
                state.nan_gradient_count = metrics.nan_gradient_count;

                if !metrics.gradients_applied && metrics.nan_gradient_count > 0 {
                    state.consecutive_nan_count += 1;
                    let emergency_threshold =
                        self.config.emergency_save_threshold.unwrap_or(5) as u32;
                    if state.consecutive_nan_count >= emergency_threshold
                        && !state.needs_emergency_save
                    {
                        state.needs_emergency_save = true;
                    }
                } else {
                    state.consecutive_nan_count = 0;
                }

                state.epoch_loss_sum += metrics.loss;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }
        }

        let step_metrics = EngineStepMetrics {
            step: metrics.step,
            loss: metrics.loss,
            mean_reward,
            std_reward,
            mean_advantage: metrics.mean_advantage,
            std_advantage: metrics.std_advantage,
            total_tokens: metrics.total_tokens,
            gradients_applied: metrics.gradients_applied,
            generation_time_ms,
            training_time_ms,
            peak_memory_mb: metrics.peak_memory_mb,
            active_memory_mb: metrics.active_memory_mb,
        };

        Ok(TrainStepResultWithOutputs {
            metrics: step_metrics,
            completions: all_completion_texts,
            rewards,
            outputs_json: outputs_json_for_return,
            completion_lengths: all_token_counts.iter().map(|&x| x as i32).collect(),
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

    /// Reset the engine for a fresh training run.
    ///
    /// This is a TERMINAL operation on this handle. It drops the training
    /// state (optimizer, step counter) on the model thread so a fresh
    /// `GRPOTrainingEngine` can be constructed on the same model, and marks
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
                .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;
            if state.invalidated {
                return Ok(());
            }
        }

        // Drop model-thread training state first (optimizer + ts.step).
        self.dispatch_reset_training_blocking()?;

        let mut state = self
            .state
            .write()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state lock"))?;

        // Preserve-and-bump the lifecycle generation so any in-flight
        // `train_step*()` with a stale `start_generation` skips its
        // post-await writeback and doesn't resurrect cleared state.
        let next_generation = state.generation.wrapping_add(1);
        *state = EngineState::default();
        state.generation = next_generation;
        state.invalidated = true;

        info!("Training engine reset (engine handle invalidated)");
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

    /// Save optimizer state (moment tensors + step) to a SafeTensors file.
    ///
    /// Routes through the model thread so AdamW moments and step counter
    /// survive across checkpoint/resume. No-op if the engine uses SGD
    /// (no optimizer to save) or before the first optimizer update has
    /// populated any moment tensors.
    #[napi]
    pub async fn save_optimizer_state(&self, path: String) -> Result<()> {
        self.ensure_valid()?;
        self.dispatch_save_optimizer_state(path).await
    }

    /// Load optimizer state (moment tensors + step) from a SafeTensors file.
    ///
    /// Routes through the model thread. No-op if the engine uses SGD.
    #[napi]
    pub async fn load_optimizer_state(&self, path: String) -> Result<()> {
        self.ensure_valid()?;
        self.dispatch_load_optimizer_state(path).await
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
                "GRPOTrainingEngine handle has been invalidated by reset(). \
                 Construct a new engine to continue training.",
            ));
        }
        Ok(())
    }

    /// Atomically check that this engine has not been invalidated and
    /// snapshot the current lifecycle generation. Used by the train_step*()
    /// methods to gate their post-await writebacks.
    fn ensure_valid_snapshot_generation(&self) -> Result<u64> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire state read lock"))?;
        if state.invalidated {
            return Err(Error::new(
                Status::GenericFailure,
                "GRPOTrainingEngine handle has been invalidated by reset(). \
                 Construct a new engine to continue training.",
            ));
        }
        Ok(state.generation)
    }
}

// =============================================================================
// Dispatch helper methods (private, not exposed to NAPI)
// =============================================================================

impl GRPOTrainingEngine {
    /// Send GenerateForTraining command and await plain data result.
    async fn dispatch_generate(
        &self,
        prompts: &[Vec<ChatMessage>],
        group_size: usize,
        gen_config: GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<GenerationPlainData> {
        // ChatMessage doesn't implement Clone (contains Uint8Array), so we
        // reconstruct each message from its serializable fields. Images are
        // not used during training, so we drop them.
        let owned_prompts: Vec<Vec<ChatMessage>> = prompts
            .iter()
            .map(|prompt_messages| {
                prompt_messages
                    .iter()
                    .map(|m| ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        tool_calls: m.tool_calls.clone(),
                        tool_call_id: m.tool_call_id.clone(),
                        reasoning_content: m.reasoning_content.clone(),
                        images: None, // Images not used in training
                    })
                    .collect()
            })
            .collect();
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::GenerateForTraining {
                        prompts: owned_prompts,
                        group_size,
                        gen_config,
                        enable_thinking,
                        tools,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::GenerateForTraining {
                        prompts: owned_prompts,
                        group_size,
                        gen_config,
                        enable_thinking,
                        tools,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::GenerateForTraining {
                        prompts: owned_prompts,
                        group_size,
                        gen_config,
                        enable_thinking,
                        tools,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Send TrainStepGRPO command and await metrics.
    async fn dispatch_train_step(
        &self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<TrainStepPlainMetrics> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::TrainStepGRPO {
                        rewards,
                        group_size,
                        loss_config,
                        valid_indices,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::TrainStepGRPO {
                        rewards,
                        group_size,
                        loss_config,
                        valid_indices,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::TrainStepGRPO {
                        rewards,
                        group_size,
                        loss_config,
                        valid_indices,
                        reply: tx,
                    })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Send BumpSkippedStep command and await the new step.
    ///
    /// Used by `train_step_auto`'s early-return paths (all-filtered and
    /// no-valid-completions). This both drops stale MxArrays on the model
    /// thread AND increments the model-thread-owned `ts.step`, returning
    /// it so the engine's read-through cache can track it.
    ///
    /// `ts.step` is the single source of truth for the training step; the
    /// engine's `EngineState.step` is only ever updated FROM the model
    /// thread (success path uses `metrics.step`, skip path uses this).
    async fn dispatch_bump_skipped_step(&self) -> Result<i64> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::BumpSkippedStep { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::BumpSkippedStep { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::BumpSkippedStep { reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Send ResetTraining command and block until complete.
    ///
    /// Drops the training state (optimizer + step counter) on the model
    /// thread so `InitTraining` can be called again. Blocking because it's
    /// invoked from the sync `reset()` NAPI method.
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

    async fn dispatch_save_optimizer_state(&self, path: String) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::SaveOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::SaveOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::SaveOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    async fn dispatch_load_optimizer_state(&self, path: String) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match &self.dispatch {
            TrainingDispatch::Qwen3(sender) => {
                sender
                    .send(Qwen3Cmd::LoadOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Dense(sender) => {
                sender
                    .send(Qwen35Cmd::LoadOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
            TrainingDispatch::Qwen35Moe(sender) => {
                sender
                    .send(Qwen35MoeCmd::LoadOptimizerState { path, reply: tx })
                    .map_err(|_| Error::from_reason("Model thread exited"))?;
            }
        }
        rx.await
            .map_err(|_| Error::from_reason("Model thread exited"))?
    }

    /// Build generation config from engine config.
    fn build_gen_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.config.max_completion_length,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            min_p: None,
            repetition_penalty: self.config.repetition_penalty,
            repetition_context_size: Some(256),
            presence_penalty: self.config.presence_penalty,
            presence_context_size: None,
            frequency_penalty: self.config.frequency_penalty,
            frequency_context_size: None,
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(self.model_type.eos_token_id()),
            return_logprobs: Some(true),
            prefill_step_size: None,
            kv_cache_bits: None,
            kv_cache_group_size: None,
            num_draft_tokens: None,
            report_performance: None,
        }
    }

    /// Build loss config from engine config.
    fn build_loss_config(&self, num_prompts: usize, group_size: usize) -> GRPOLossConfig {
        GRPOLossConfig {
            epsilon_low: self.config.clip_epsilon.unwrap_or(0.2),
            epsilon_high: None,
            beta: self.config.kl_coef.unwrap_or(0.0),
            loss_type: self
                .config
                .loss_type
                .clone()
                .unwrap_or_else(|| "grpo".to_string()),
            importance_sampling_level: "token".to_string(),
            max_completion_length: self.config.max_completion_length.map(|n| n as i64),
            num_items_in_batch: Some((num_prompts * group_size) as f64),
            gradient_accumulation_steps: self.config.gradient_accumulation_steps.unwrap_or(1)
                as i64,
            lm_head_chunk_size: self.config.lm_head_chunk_size.map(|n| n as i64),
            forward_chunk_size: self.config.forward_chunk_size.map(|n| n as i64),
            vocab_chunk_size: self.config.vocab_chunk_size.map(|n| n as i64),
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

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

/// Compute per-prompt balanced valid indices for GRPO degenerate-output filtering.
///
/// The cache layout is prompt-major: completion index `c` belongs to prompt
/// `c / group_size`. After filtering "length"-degenerate completions per-prompt,
/// we take the FIRST `min_survivor_count` survivors from each prompt so that the
/// result is rectangular (num_prompts * effective_group_size). This keeps the
/// divisibility invariant required by advantages.rs (`rewards.len() % group_size == 0`)
/// and autograd.rs (`num_prompts * group_size` completions).
///
/// # Arguments
/// * `all_finish_reasons` - finish reason per completion (length == num_prompts * group_size)
/// * `all_token_counts`   - token count per completion (length == num_prompts * group_size)
/// * `num_prompts`        - number of prompts in this batch
/// * `group_size`         - original completions-per-prompt before filtering
/// * `max_tokens_threshold` - drop a completion if finish_reason == "length" and tokens >= this
///
/// # Returns
/// `(balanced_valid_indices, effective_group_size, per_prompt_survivor_counts)`
/// * `balanced_valid_indices` - rectangular, prompt-major list of kept indices
/// * `effective_group_size` - min survivors across all prompts (0 if any prompt lost all)
/// * `per_prompt_survivor_counts` - raw per-prompt survivor counts, pre-balancing (for logging)
pub(crate) fn compute_balanced_valid_indices(
    all_finish_reasons: &[String],
    all_token_counts: &[u32],
    num_prompts: usize,
    group_size: usize,
    max_tokens_threshold: u32,
) -> (Vec<usize>, usize, Vec<usize>) {
    if num_prompts == 0 || group_size == 0 {
        return (Vec::new(), 0, Vec::new());
    }

    // Collect survivors per prompt, preserving prompt-major order.
    let mut per_prompt: Vec<Vec<usize>> = (0..num_prompts).map(|_| Vec::new()).collect();
    for (idx, reason) in all_finish_reasons.iter().enumerate() {
        let token_count = all_token_counts.get(idx).copied().unwrap_or(0);
        let is_degenerate = reason == "length" && token_count >= max_tokens_threshold;
        if is_degenerate {
            continue;
        }
        let prompt_idx = idx / group_size;
        if prompt_idx < num_prompts {
            per_prompt[prompt_idx].push(idx);
        }
    }

    let per_prompt_counts: Vec<usize> = per_prompt.iter().map(|v| v.len()).collect();
    let effective_group_size = per_prompt_counts.iter().copied().min().unwrap_or(0);

    // Rebuild valid_indices in prompt-major order, taking the first
    // `effective_group_size` survivors from each prompt. By construction this
    // guarantees `valid_indices.len() == num_prompts * effective_group_size`.
    let mut balanced_valid_indices =
        Vec::with_capacity(num_prompts.saturating_mul(effective_group_size));
    for prompt_survivors in &per_prompt {
        for &idx in prompt_survivors.iter().take(effective_group_size) {
            balanced_valid_indices.push(idx);
        }
    }

    (
        balanced_valid_indices,
        effective_group_size,
        per_prompt_counts,
    )
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

    fn mk_reasons(len: usize, degenerate_idxs: &[usize]) -> Vec<String> {
        (0..len)
            .map(|i| {
                if degenerate_idxs.contains(&i) {
                    "length".to_string()
                } else {
                    "stop".to_string()
                }
            })
            .collect()
    }

    #[test]
    fn test_balanced_filter_no_filtering() {
        // 3 prompts * 4 completions = 12, none degenerate
        let reasons = mk_reasons(12, &[]);
        let tokens = vec![10u32; 12];
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 3, 4, 100);
        assert_eq!(eff, 4);
        assert_eq!(valid.len(), 3 * 4);
        assert_eq!(valid, (0..12).collect::<Vec<_>>());
        assert_eq!(counts, vec![4, 4, 4]);
    }

    #[test]
    fn test_balanced_filter_uniform_drop() {
        // 3 prompts * 4 = 12, drop 1 from each prompt: indices 0, 4, 8
        // All those have finish_reason=length AND token_count >= threshold
        let reasons = mk_reasons(12, &[0, 4, 8]);
        let mut tokens = vec![10u32; 12];
        tokens[0] = 200;
        tokens[4] = 200;
        tokens[8] = 200;
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 3, 4, 100);
        assert_eq!(eff, 3);
        assert_eq!(valid.len(), 3 * 3);
        // Prompt 0: survivors [1,2,3], prompt 1: [5,6,7], prompt 2: [9,10,11]
        assert_eq!(valid, vec![1, 2, 3, 5, 6, 7, 9, 10, 11]);
        assert_eq!(counts, vec![3, 3, 3]);
    }

    #[test]
    fn test_balanced_filter_uneven_drop() {
        // 3 prompts * 4 = 12
        // Prompt 0 loses 1 (idx 0), prompt 1 loses 3 (idxs 4,5,6), prompt 2 loses 0
        let reasons = mk_reasons(12, &[0, 4, 5, 6]);
        let mut tokens = vec![10u32; 12];
        for &i in &[0usize, 4, 5, 6] {
            tokens[i] = 200;
        }
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 3, 4, 100);
        // min survivors = 1 (prompt 1 kept only idx 7)
        assert_eq!(eff, 1);
        assert_eq!(counts, vec![3, 1, 4]);
        assert_eq!(valid.len(), 3);
        // prompt 0 first survivor = 1, prompt 1 = 7, prompt 2 = 8
        assert_eq!(valid, vec![1, 7, 8]);
    }

    #[test]
    fn test_balanced_filter_all_from_one_prompt() {
        // 3 prompts * 4 = 12; prompt 1 loses ALL 4 completions (idxs 4-7)
        let reasons = mk_reasons(12, &[4, 5, 6, 7]);
        let mut tokens = vec![10u32; 12];
        for t in tokens.iter_mut().skip(4).take(4) {
            *t = 200;
        }
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 3, 4, 100);
        assert_eq!(eff, 0);
        assert_eq!(valid.len(), 0);
        assert_eq!(counts, vec![4, 0, 4]);
    }

    #[test]
    fn test_balanced_filter_asymmetric_whole_loss() {
        // 2 prompts * 3 = 6; prompt 0 keeps all, prompt 1 loses all
        let reasons = mk_reasons(6, &[3, 4, 5]);
        let mut tokens = vec![10u32; 6];
        for t in tokens.iter_mut().skip(3).take(3) {
            *t = 200;
        }
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 2, 3, 100);
        assert_eq!(eff, 0);
        assert_eq!(valid.len(), 0);
        assert_eq!(counts, vec![3, 0]);
    }

    #[test]
    fn test_balanced_filter_rectangular_invariant() {
        // Random-ish but deterministic: 4 prompts * 5 = 20
        // prompt 0 loses 0, prompt 1 loses 2 (idxs 6, 8), prompt 2 loses 1 (idx 11), prompt 3 loses 4 (15,16,17,18)
        let degenerate = vec![6usize, 8, 11, 15, 16, 17, 18];
        let reasons = mk_reasons(20, &degenerate);
        let mut tokens = vec![10u32; 20];
        for &i in &degenerate {
            tokens[i] = 200;
        }
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 4, 5, 100);
        // Survivor counts: [5, 3, 4, 1] → min = 1
        assert_eq!(counts, vec![5, 3, 4, 1]);
        assert_eq!(eff, 1);
        // Rectangular: 4 * 1 = 4
        assert_eq!(valid.len(), 4);
        // First survivor of each prompt: 0 (p0), 5 (p1 first non-degen), 10 (p2 first non-degen), 19 (p3 only survivor)
        assert_eq!(valid, vec![0, 5, 10, 19]);
        // Verify divisibility
        assert_eq!(valid.len() % 4, 0);
    }

    #[test]
    fn test_balanced_filter_length_but_under_threshold() {
        // finish_reason == "length" but token_count < threshold → NOT degenerate
        let reasons = mk_reasons(6, &[0, 1, 2, 3, 4, 5]);
        let tokens = vec![50u32; 6]; // below threshold 100
        let (valid, eff, counts) = compute_balanced_valid_indices(&reasons, &tokens, 2, 3, 100);
        assert_eq!(eff, 3);
        assert_eq!(valid.len(), 6);
        assert_eq!(counts, vec![3, 3]);
    }

    #[test]
    fn test_balanced_filter_empty_inputs() {
        let (valid, eff, counts) = compute_balanced_valid_indices(&[], &[], 0, 4, 100);
        assert_eq!(eff, 0);
        assert!(valid.is_empty());
        assert!(counts.is_empty());
    }
}
