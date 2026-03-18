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
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::{
    MxArray, get_active_memory, get_peak_memory, heavy_cleanup, reset_peak_memory,
    synchronize_and_clear_cache,
};
use crate::grpo::advantages::compute_advantages;
use crate::grpo::autograd::compute_loss_and_gradients_autograd;
use crate::grpo::loss::GRPOLossConfig;
use crate::grpo::rewards::{
    BuiltinRewardConfig, JsonSchemaReward, LengthReward, RewardRegistry, ToolUseReward,
    XMLFormatReward,
};
use crate::models::qwen3::{GenerationConfig, Qwen3Model};
use crate::models::qwen3_5::model::Qwen3_5Model;
use crate::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use crate::optimizers::GradientUtils;
use crate::tokenizer::{ChatMessage, ToolDefinition};
use crate::tools::build_reward_outputs;
use crate::training_model::{ModelType, TrainableModel, TrainableModelEnum};

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
    model: Arc<RwLock<TrainableModelEnum>>,
    /// Model type (carries config for functional forward pass in autograd)
    model_type: ModelType,
    /// Engine configuration
    config: GRPOEngineConfig,
    /// Reward registry (built-in rewards)
    reward_registry: RewardRegistry,
    /// Training state
    state: Arc<RwLock<EngineState>>,
    /// AdamW optimizer (used when optimizer_type is "adamw")
    optimizer: Option<Arc<std::sync::Mutex<crate::optimizers::AdamW>>>,
}

/// Apply gradients to model using either AdamW or SGD.
fn apply_optimizer_step(
    model: &mut TrainableModelEnum,
    grads: &HashMap<String, MxArray>,
    params: &HashMap<String, MxArray>,
    lr: f64,
    optimizer: &Option<Arc<std::sync::Mutex<crate::optimizers::AdamW>>>,
    grad_acc_steps: i32,
) -> Result<()> {
    if let Some(opt_arc) = optimizer {
        // AdamW path
        let mut opt = opt_arc
            .lock()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire optimizer lock"))?;

        let mut param_names_vec: Vec<String> = Vec::new();
        let mut param_refs: Vec<&MxArray> = Vec::new();
        let mut grad_refs: Vec<&MxArray> = Vec::new();

        // When using gradient accumulation, gradients are summed (not averaged).
        // SGD compensates by dividing lr, but AdamW ignores lr (uses delta trick with 1.0).
        // So we must average the gradients explicitly before passing to AdamW.
        let scaled_grads: HashMap<String, MxArray>;
        let grads_to_use = if grad_acc_steps > 1 {
            let scale = 1.0 / grad_acc_steps as f32;
            let scale_arr = MxArray::from_float32(&[scale], &[]).unwrap();
            scaled_grads = grads
                .iter()
                .map(|(name, grad)| (name.clone(), grad.mul(&scale_arr).unwrap()))
                .collect();
            &scaled_grads
        } else {
            grads
        };

        for (name, grad) in grads_to_use {
            if let Some(param) = params.get(name) {
                param_names_vec.push(name.clone());
                param_refs.push(param);
                grad_refs.push(grad);
            }
        }

        let updated = opt.update_batch(param_names_vec.clone(), param_refs.clone(), grad_refs)?;

        // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
        let deltas: HashMap<String, MxArray> = param_names_vec
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let delta = param_refs[i].sub(&updated[i]).unwrap();
                (name.clone(), delta)
            })
            .collect();

        let delta_refs: HashMap<String, &MxArray> =
            deltas.iter().map(|(k, v)| (k.clone(), v)).collect();
        model.apply_gradients_with_params(delta_refs, 1.0, params)?;

        debug!("Applied AdamW update (step={})", opt.get_step());
    } else {
        // SGD path
        let grads_refs: HashMap<String, &MxArray> =
            grads.iter().map(|(k, v)| (k.clone(), v)).collect();
        model.apply_gradients_with_params(grads_refs, lr, params)?;

        debug!("Applied SGD gradients with lr: {}", lr);
    }

    Ok(())
}

/// Create an AdamW optimizer from engine config, or None for SGD.
fn create_optimizer(
    config: &GRPOEngineConfig,
) -> Option<Arc<std::sync::Mutex<crate::optimizers::AdamW>>> {
    let opt_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    if opt_type == "adamw" {
        let optimizer = crate::optimizers::AdamW::new(
            config.learning_rate,
            config.adamw_beta1,
            config.adamw_beta2,
            config.adamw_eps,
            config.weight_decay,
            Some(true), // bias correction
        );
        info!(
            "Using AdamW optimizer (lr={}, beta1={}, beta2={}, wd={})",
            config.learning_rate.unwrap_or(1e-6),
            config.adamw_beta1.unwrap_or(0.9),
            config.adamw_beta2.unwrap_or(0.999),
            config.weight_decay.unwrap_or(0.01),
        );
        Some(Arc::new(std::sync::Mutex::new(optimizer)))
    } else {
        info!(
            "Using SGD optimizer (lr={})",
            config.learning_rate.unwrap_or(1e-6)
        );
        None
    }
}

#[napi]
impl GRPOTrainingEngine {
    /// Create a new training engine from a Qwen3 model
    ///
    /// # Arguments
    /// * `model` - The Qwen3 model to train (will be cloned internally)
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

        let optimizer = create_optimizer(&config);

        Ok(Self {
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen3(
                model.clone_for_session()?,
            ))),
            model_type,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
            optimizer,
        })
    }

    /// Create a new training engine from a Qwen3.5 dense model
    #[napi(factory)]
    pub fn from_qwen35(model: &Qwen3_5Model, config: GRPOEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Dense(model.get_config());

        info!(
            "Creating training engine (Qwen3.5 Dense): {} layers, {} hidden",
            model_type.num_layers(),
            model_type.hidden_size()
        );

        let optimizer = create_optimizer(&config);

        Ok(Self {
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen35Dense(
                model.clone_for_training()?,
            ))),
            model_type,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
            optimizer,
        })
    }

    /// Create a new training engine from a Qwen3.5 MoE model
    #[napi(factory)]
    pub fn from_qwen35_moe(model: &Qwen3_5MoeModel, config: GRPOEngineConfig) -> Result<Self> {
        let model_type = ModelType::Qwen35Moe(model.get_config());

        info!(
            "Creating training engine (Qwen3.5 MoE): {} layers, {} hidden",
            model_type.num_layers(),
            model_type.hidden_size()
        );

        let optimizer = create_optimizer(&config);

        Ok(Self {
            model: Arc::new(RwLock::new(TrainableModelEnum::Qwen35Moe(
                model.clone_for_training()?,
            ))),
            model_type,
            config,
            reward_registry: RewardRegistry::new(),
            state: Arc::new(RwLock::new(EngineState::default())),
            optimizer,
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
        let optimizer_arc = self.optimizer.clone();
        let model_type = self.model_type.clone();
        let config = self.config.clone();
        let enable_thinking = config.enable_thinking;
        let tools = config.tools.clone();

        // Build generation config - use model's eos_token_id explicitly
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_completion_length,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_type.eos_token_id()),
            return_logprobs: Some(true),
            prefill_step_size: None, // Use default (2048)
            kv_cache_bits: None,     // Default: no quantization
            kv_cache_group_size: None,
            num_draft_tokens: None, // Speculative decoding not used in GRPO
            report_performance: None,
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
                // Tokenize prompt with tools for proper tool calling format
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(
                        &prompt_messages,
                        Some(true),
                        tools.as_deref(),
                        enable_thinking,
                    )?
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

                    // CRITICAL: Use heavy_cleanup() to clear KV cache, intermediate tensors,
                    // AND compiler cache after each completion. This prevents Metal context
                    // accumulation that causes "Context leak detected" warnings.
                    // The compiler cache holds Metal command buffers that can accumulate.
                    heavy_cleanup();
                }
            }

            let generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;

            // Heavy cleanup after generation phase to release ALL Metal contexts
            // This is more aggressive than synchronize_and_clear_cache() and helps
            // prevent Metal driver context leaks.
            heavy_cleanup();

            let training_start = std::time::Instant::now();
            reset_peak_memory(); // Reset peak memory counter for this step

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
                max_completion_length: config.max_completion_length.map(|n| n as i64),
                num_items_in_batch: Some(
                    (num_prompts * config.group_size.unwrap_or(4) as usize) as f64,
                ),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
                lm_head_chunk_size: config.lm_head_chunk_size.map(|n| n as i64),
                forward_chunk_size: config.forward_chunk_size.map(|n| n as i64),
                vocab_chunk_size: config.vocab_chunk_size.map(|n| n as i64),
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()?
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens_all.iter().collect();
            let completion_refs: Vec<&MxArray> = completion_tokens_all.iter().collect();
            let logprob_refs: Vec<&MxArray> = completion_logprobs_all.iter().collect();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_type,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards,
                config.group_size.unwrap_or(4),
                loss_config,
                config.gradient_checkpointing.unwrap_or(true),
            )?;

            // Check for NaN
            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                let total_tokens: i32 = token_counts_all.iter().sum();

                let state = state_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state read lock")
                })?;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                });
            }

            // Step 1: Validate ALL gradients first - if ANY has NaN/Inf, skip entire step
            // This prevents partial gradient application which can degrade model weights
            // Uses GPU-native has_nan_or_inf() to avoid transferring entire gradient tensors to CPU
            // (For Qwen3-0.6B: transfers ~4 bytes per gradient instead of ~2.4GB total)
            let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
            for (name, grad) in gradients.iter() {
                grad.eval();
                // GPU-native check: only transfers a single boolean to CPU
                let has_invalid = grad.has_nan_or_inf()?;
                if has_invalid {
                    // Only do detailed CPU analysis in verbose mode (for debugging)
                    let invalid_count = if verbose_nan {
                        let data = grad.to_float32()?;
                        data.iter()
                            .filter(|v| v.is_nan() || v.is_infinite())
                            .count()
                    } else {
                        0 // Unknown count in fast mode
                    };

                    if verbose_nan {
                        warn!(
                            "Gradient '{}' contains {} invalid values (NaN/Inf) - SKIPPING ENTIRE STEP to prevent model corruption",
                            name, invalid_count
                        );
                    } else {
                        warn!(
                            "Gradient '{}' contains NaN/Inf values - SKIPPING ENTIRE STEP to prevent model corruption (enable verbose_nan_detection for counts)",
                            name
                        );
                    }

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
                        std_advantage: 0.0,
                        total_tokens,
                        gradients_applied: false,
                        generation_time_ms,
                        training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                        peak_memory_mb: get_peak_memory() / 1e6,
                        active_memory_mb: get_active_memory() / 1e6,
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
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;

            // Accumulate gradients with NaN/Inf checking
            let acc_result = accumulate_gradients(&mut state, gradients)?;

            // Handle invalid gradients found during accumulation
            if acc_result.had_invalid_gradients {
                state.consecutive_nan_count += 1;
                state.nan_gradient_count += acc_result.invalid_param_count as u64;
                warn!(
                    "Found {} parameters with NaN/Inf during accumulation: {:?} (consecutive: {})",
                    acc_result.invalid_param_count,
                    acc_result.invalid_param_names,
                    state.consecutive_nan_count
                );

                // Check emergency save threshold
                let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                if state.consecutive_nan_count >= emergency_threshold && !state.needs_emergency_save
                {
                    state.needs_emergency_save = true;
                    warn!(
                        "Emergency save triggered: {} consecutive steps with NaN/Inf gradients",
                        state.consecutive_nan_count
                    );
                }
            } else {
                // Reset consecutive NaN count on fully successful accumulation
                state.consecutive_nan_count = 0;
            }

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

                apply_optimizer_step(&mut model_mut, &grads, &params, lr, &optimizer_arc, grad_acc_steps)?;

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
            let std_advantage = {
                let variance = adv_data
                    .iter()
                    .map(|&a| {
                        let diff = a as f64 - mean_advantage;
                        diff * diff
                    })
                    .sum::<f64>()
                    / adv_data.len() as f64;
                variance.sqrt()
            };

            // CRITICAL: Always call heavy_cleanup after autograd to clear compiled graph cache.
            // Without compile_clear_cache(), the C++ side accumulates cached compiled graphs,
            // causing unbounded memory growth. This was the root cause of OOM issues.
            heavy_cleanup();

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
                std_advantage,
                total_tokens,
                gradients_applied,
                generation_time_ms,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
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
        let model_type = self.model_type.clone();
        let enable_thinking = config.enable_thinking;
        let tools = config.tools.clone();

        // Build generation config - use model's eos_token_id explicitly
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_completion_length,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_type.eos_token_id()),
            return_logprobs: Some(true),
            prefill_step_size: None, // Use default (2048)
            kv_cache_bits: None,     // Default: no quantization
            kv_cache_group_size: None,
            num_draft_tokens: None, // Speculative decoding not used in GRPO
            report_performance: None,
        };

        let result = napi::bindgen_prelude::spawn_blocking(move || {
            let num_completions = num_prompts * group_size;
            let max_tokens = gen_config.max_new_tokens.unwrap_or(256) as usize;

            // Step 1: Tokenize all prompts first
            let mut prompt_arrays: Vec<MxArray> = Vec::with_capacity(num_prompts);
            for prompt_messages in &prompts {
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(
                        prompt_messages,
                        Some(true),
                        tools.as_deref(),
                        enable_thinking,
                    )?
                };
                let prompt_array =
                    MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
                prompt_arrays.push(prompt_array);
            }

            // Step 2: Batched generation
            // Use parallel batch generation if enabled (true batch with per-sequence RoPE offsets)
            // Otherwise use sequential (prefill once per prompt, batch decode G completions)
            let use_parallel = config.use_parallel_batch_generation.unwrap_or(false);
            let batch_result = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                if use_parallel {
                    // Parallel batch generation is only available for Qwen3 models
                    match &*model {
                        TrainableModelEnum::Qwen3(m) => {
                            m.generate_batch_parallel_sync(&prompt_arrays, group_size, Some(gen_config.clone()))?
                        }
                        _ => {
                            // Fall back to sequential for non-Qwen3 models
                            model.generate_batch_for_training_sync(&prompt_arrays, group_size, Some(gen_config.clone()))?
                        }
                    }
                } else {
                    model.generate_batch_for_training_sync(&prompt_arrays, group_size, Some(gen_config.clone()))?
                }
            };

            // Step 3: Decode all completions and convert to expected format
            let mut completion_texts: Vec<String> = Vec::with_capacity(num_completions);
            let mut all_tokens: Vec<i64> = Vec::with_capacity(num_completions * max_tokens);
            let mut all_logprobs: Vec<f64> = Vec::with_capacity(num_completions * max_tokens);
            let mut completion_lengths: Vec<i32> = Vec::with_capacity(num_completions);
            let mut finish_reasons: Vec<String> = Vec::with_capacity(num_completions);

            // Results are ordered: [prompt0_comp0, prompt0_comp1, ..., prompt1_comp0, ...]
            for (i, tokens_arr) in batch_result.tokens.iter().enumerate() {
                // Decode tokens to text
                let text = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.decode_tokens_sync(tokens_arr)?
                };
                completion_texts.push(text);

                // Extract token IDs and logprobs
                let tokens = tokens_arr.to_int32()?;
                let logprobs = batch_result.logprobs[i].to_float32()?;

                completion_lengths.push(tokens.len() as i32);
                all_tokens.extend(tokens.iter().map(|&x| x as i64));
                all_logprobs.extend(logprobs.iter().map(|&x| x as f64));

                // Get finish reason from batch result
                // Use batch_result's group_size to ensure consistent indexing
                let result_group_size = batch_result.group_size as usize;
                let prompt_idx = i / result_group_size;
                let group_idx = i % result_group_size;

                // Bounds check with helpful error message
                let reason = batch_result.finish_reasons
                    .get(prompt_idx)
                    .and_then(|reasons: &Vec<String>| reasons.get(group_idx))
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!(
                            "WARN: finish_reasons out of bounds - i={}, prompt_idx={}, group_idx={}, \
                             finish_reasons.len()={}, result_group_size={}, tokens.len()={}",
                            i, prompt_idx, group_idx,
                            batch_result.finish_reasons.len(), result_group_size, batch_result.tokens.len()
                        );
                        "unknown".to_string()
                    });
                finish_reasons.push(reason);
            }

            // Heavy cleanup after generation phase to release ALL Metal contexts
            heavy_cleanup();
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
        reset_peak_memory(); // Reset peak memory counter for this step

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let optimizer_arc = self.optimizer.clone();
        let model_type = self.model_type.clone();
        let config = self.config.clone();
        let enable_thinking = config.enable_thinking;
        let tools = config.tools.clone();

        // Run the training step in spawn_blocking
        let metrics = napi::bindgen_prelude::spawn_blocking(move || {
            // === Phase 1: Tokenize prompts and reconstruct completion arrays ===
            let mut prompt_tokens_all: Vec<MxArray> = Vec::with_capacity(num_prompts);

            for prompt_messages in prompts {
                // Tokenize prompt with tools for proper tool calling format
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(
                        &prompt_messages,
                        Some(true),
                        tools.as_deref(),
                        enable_thinking,
                    )?
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
                max_completion_length: config.max_completion_length.map(|n| n as i64),
                num_items_in_batch: Some(expected_rewards as f64),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
                lm_head_chunk_size: config.lm_head_chunk_size.map(|n| n as i64),
                forward_chunk_size: config.forward_chunk_size.map(|n| n as i64),
                vocab_chunk_size: config.vocab_chunk_size.map(|n| n as i64),
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()?
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens_all.iter().collect();
            let completion_refs: Vec<&MxArray> = completion_tokens_all.iter().collect();
            let logprob_refs: Vec<&MxArray> = completion_logprobs_all.iter().collect();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_type,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards,
                config.group_size.unwrap_or(4),
                loss_config,
                config.gradient_checkpointing.unwrap_or(true),
            )?;

            // Check for NaN
            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards);
                let total_tokens: i32 = token_counts_all.iter().sum();

                let state = state_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state read lock")
                })?;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms: 0.0, // Not measured here, was done separately
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                });
            }

            // Step 1: Validate and clamp gradient values to prevent extreme values
            // This happens BEFORE norm clipping, as Inf values break norm computation
            let grad_clip_value = config.gradient_clip_value.unwrap_or(1.0);
            let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
            let mut has_invalid_grad = false;
            let mut invalid_param_names: Vec<String> = Vec::new();

            for (name, grad) in gradients.iter() {
                grad.eval();

                // GPU-native check: transfers only ~4 bytes instead of entire gradient tensor
                if grad.has_nan_or_inf()? {
                    warn!(
                        "Gradient '{}' contains NaN/Inf values - will skip step",
                        name
                    );
                    has_invalid_grad = true;
                    invalid_param_names.push(name.clone());
                    continue; // Don't insert anything - skip this gradient entirely
                }

                // Clamp to reasonable range (lazy - let MLX fuse operations)
                let clamped = grad.clip(Some(-grad_clip_value), Some(grad_clip_value))?;
                clamped_gradients.insert(name.clone(), clamped);
            }

            // Log all invalid parameters if any were found
            if !invalid_param_names.is_empty() {
                warn!(
                    "Found {} parameters with NaN/Inf gradients: {:?}",
                    invalid_param_names.len(),
                    invalid_param_names
                );
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

                // Track NaN gradient occurrences (step NOT incremented on NaN skip)
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
                    std_advantage: 0.0,
                    total_tokens,
                    gradients_applied: false,
                    generation_time_ms: 0.0,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                });
            }

            // === Phase 3: Accumulate and apply gradients ===
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;

            // Accumulate gradients with NaN/Inf checking
            let acc_result = accumulate_gradients(&mut state, gradients)?;

            // Handle invalid gradients found during accumulation
            if acc_result.had_invalid_gradients {
                state.consecutive_nan_count += 1;
                state.nan_gradient_count += acc_result.invalid_param_count as u64;
                warn!(
                    "Found {} parameters with NaN/Inf during accumulation: {:?} (consecutive: {})",
                    acc_result.invalid_param_count,
                    acc_result.invalid_param_names,
                    state.consecutive_nan_count
                );

                // Check emergency save threshold
                let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                if state.consecutive_nan_count >= emergency_threshold && !state.needs_emergency_save
                {
                    state.needs_emergency_save = true;
                    warn!(
                        "Emergency save triggered: {} consecutive steps with NaN/Inf gradients",
                        state.consecutive_nan_count
                    );
                }
            } else {
                // Reset consecutive NaN count on fully successful accumulation
                state.consecutive_nan_count = 0;
            }

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

                apply_optimizer_step(&mut model_mut, &grads, &params, lr, &optimizer_arc, grad_acc_steps)?;

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

            // Compute mean and std advantage
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
            let std_advantage = {
                let variance = adv_data
                    .iter()
                    .map(|&a| {
                        let diff = a as f64 - mean_advantage;
                        diff * diff
                    })
                    .sum::<f64>()
                    / adv_data.len() as f64;
                variance.sqrt()
            };

            // CRITICAL: Always call heavy_cleanup after autograd to clear compiled graph cache.
            // Without compile_clear_cache(), the C++ side accumulates cached compiled graphs,
            // causing unbounded memory growth. This was the root cause of OOM issues.
            heavy_cleanup();

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
                std_advantage,
                total_tokens,
                gradients_applied,
                generation_time_ms: 0.0,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
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

    /// Unified training step with JS reward callback and optional output recording
    ///
    /// Same as `train_step_auto` but optionally captures the full RewardOutput data
    /// for persistence to an output store database.
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

        // Clone Arcs for the blocking task
        let model_arc = Arc::clone(&self.model);
        let model_type = self.model_type.clone();
        let config = self.config.clone();
        let enable_thinking = config.enable_thinking;
        let tools = config.tools.clone();

        // Build generation config
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_completion_length,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: Some(256),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: Some(model_type.eos_token_id()),
            return_logprobs: Some(true),
            prefill_step_size: None, // Use default (2048)
            kv_cache_bits: None,     // Default: no quantization
            kv_cache_group_size: None,
            num_draft_tokens: None, // Speculative decoding not used in GRPO
            report_performance: None,
        };

        // === Phase 1: Generate completions ===
        info!(
            "Phase 1: Generating {} completions ({} prompts × {} groups)",
            expected_completions, num_prompts, group_size
        );
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
                // Tokenize prompt with tools for proper tool calling format
                let prompt_token_ids = {
                    let model = model_arc.read().map_err(|_| {
                        Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                    })?;
                    model.apply_chat_template_sync(
                        &prompt_messages,
                        Some(true),
                        tools.as_deref(),
                        enable_thinking,
                    )?
                };

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

                for _g in 0..group_size {
                    let result = {
                        let model = model_arc.read().map_err(|_| {
                            Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                        })?;
                        model.generate_for_training_sync(&prompt_2d, Some(gen_config.clone()))?
                    };

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
                }
            }

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
        info!(
            "Phase 1 complete: generated in {:.1}s",
            generation_time_ms / 1000.0
        );

        let IntermediateGenerationResult {
            completion_texts,
            prompt_texts,
            prompt_tokens,
            completion_tokens,
            completion_logprobs,
            token_counts,
            finish_reasons,
        } = gen_result;

        let finish_reasons_for_filter = finish_reasons.clone();

        // === Phase 2: Build RewardOutput[] and call JS reward function ===
        info!("Phase 2: Computing rewards via JS callback...");
        let reward_outputs = build_reward_outputs(
            prompt_texts,
            completion_texts.clone(),
            token_counts.clone(),
            finish_reasons,
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

        // === DEGENERATE OUTPUT FILTERING ===
        let max_tokens_threshold =
            (self.config.max_completion_length.unwrap_or(4096) as f64 * 0.9) as u32;
        let valid_indices: Vec<usize> = finish_reasons_for_filter
            .iter()
            .enumerate()
            .filter(|(i, reason)| *reason != "length" || token_counts[*i] < max_tokens_threshold)
            .map(|(i, _)| i)
            .collect();

        let num_filtered = expected_completions - valid_indices.len();
        if num_filtered > 0 {
            info!(
                "Filtered {} degenerate completions (finish_reason='length', tokens >= {})",
                num_filtered, max_tokens_threshold
            );
        }

        if valid_indices.is_empty() {
            warn!(
                "All {} completions hit token limit - skipping training step to prevent OOM",
                expected_completions
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
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                },
                completions: completion_texts,
                rewards,
                outputs_json: outputs_json_for_return,
                completion_lengths: token_counts.iter().map(|&x| x as i32).collect(),
            });
        }

        let filtered_count = valid_indices.len();
        let effective_group_size = if num_prompts > 0 {
            filtered_count / num_prompts
        } else {
            group_size
        };

        if effective_group_size < 1 {
            warn!(
                "Only {} valid completions for {} prompts - skipping training",
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
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                },
                completions: completion_texts,
                rewards,
                outputs_json: outputs_json_for_return,
                completion_lengths: token_counts.iter().map(|&x| x as i32).collect(),
            });
        }

        let usable_count = num_prompts * effective_group_size;
        let valid_indices: Vec<usize> = if usable_count < filtered_count {
            valid_indices.into_iter().take(usable_count).collect()
        } else {
            valid_indices
        };

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

        synchronize_and_clear_cache();

        // === Phase 3: Train ===
        info!(
            "Phase 3: Training with {} valid completions (filtered {})",
            valid_indices.len(),
            num_filtered
        );
        let training_start = std::time::Instant::now();
        reset_peak_memory(); // Reset peak memory counter for this step

        let model_arc = Arc::clone(&self.model);
        let state_arc = Arc::clone(&self.state);
        let optimizer_arc = self.optimizer.clone();
        let model_type = self.model_type.clone();
        let config = self.config.clone();
        let rewards_clone = filtered_rewards.clone();
        let group_size_for_training = effective_group_size as i32;

        let metrics = napi::bindgen_prelude::spawn_blocking(move || {
            let loss_config = GRPOLossConfig {
                epsilon_low: config.clip_epsilon.unwrap_or(0.2),
                epsilon_high: None,
                beta: config.kl_coef.unwrap_or(0.0),
                loss_type: config
                    .loss_type
                    .clone()
                    .unwrap_or_else(|| "grpo".to_string()),
                importance_sampling_level: "token".to_string(),
                max_completion_length: config.max_completion_length.map(|n| n as i64),
                num_items_in_batch: Some(usable_count as f64),
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as i64,
                lm_head_chunk_size: config.lm_head_chunk_size.map(|n| n as i64),
                forward_chunk_size: config.forward_chunk_size.map(|n| n as i64),
                vocab_chunk_size: config.vocab_chunk_size.map(|n| n as i64),
            };

            let params = {
                let model = model_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire model read lock")
                })?;
                model.get_parameters()?
            };

            let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
            let completion_refs: Vec<&MxArray> = filtered_completion_tokens.iter().collect();
            let logprob_refs: Vec<&MxArray> = filtered_completion_logprobs.iter().collect();

            info!(
                "Computing loss and gradients ({} prompts, {} completions)",
                prompt_refs.len(),
                completion_refs.len()
            );
            let grad_start = std::time::Instant::now();

            let (loss_value, gradients) = compute_loss_and_gradients_autograd(
                &model_type,
                &params,
                &prompt_refs,
                &completion_refs,
                &logprob_refs,
                &rewards_clone,
                group_size_for_training,
                loss_config,
                config.gradient_checkpointing.unwrap_or(true),
            )?;

            info!(
                "Loss computed in {:.1}s: {:.4} ({} gradients)",
                grad_start.elapsed().as_secs_f64(),
                loss_value,
                gradients.len()
            );

            if loss_value.is_nan() || loss_value.is_infinite() {
                warn!("Skipping step due to invalid loss: {}", loss_value);
                synchronize_and_clear_cache();

                let (mean_reward, std_reward) = compute_reward_stats(&rewards_clone);
                let total_tokens: u32 = filtered_token_counts.iter().sum();

                let state = state_arc.read().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state read lock")
                })?;

                return Ok::<EngineStepMetrics, Error>(EngineStepMetrics {
                    step: state.step,
                    loss: loss_value,
                    mean_reward,
                    std_reward,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    total_tokens: total_tokens as i32,
                    gradients_applied: false,
                    generation_time_ms,
                    training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                });
            }

            // Validate gradients using GPU-native check (transfers only ~4 bytes per gradient)
            info!("Validating {} gradients...", gradients.len());
            let validate_start = std::time::Instant::now();
            for (name, grad) in gradients.iter() {
                grad.eval();

                // GPU-native check: more thorough than sum-based check, catches sparse NaN values
                if grad.has_nan_or_inf()? {
                    warn!(
                        "Gradient '{}' contains NaN/Inf values - SKIPPING STEP",
                        name
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
                        std_advantage: 0.0,
                        total_tokens: total_tokens as i32,
                        gradients_applied: false,
                        generation_time_ms,
                        training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                        peak_memory_mb: get_peak_memory() / 1e6,
                        active_memory_mb: get_active_memory() / 1e6,
                    });
                }
            }
            info!(
                "Gradients validated in {:.1}s",
                validate_start.elapsed().as_secs_f64()
            );

            // Clip gradients
            info!("Clipping and applying gradients...");
            let apply_start = std::time::Instant::now();
            let grad_clip_value = config.gradient_clip_value.unwrap_or(1.0);
            let grad_refs: HashMap<String, &MxArray> =
                gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
            let gradients = GradientUtils::clip_grad_value_and_norm(
                grad_refs,
                grad_clip_value,
                config.gradient_clip_norm,
            )?;

            // Apply gradients
            let mut state = state_arc.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire state write lock")
            })?;

            // Accumulate gradients with NaN/Inf checking
            let acc_result = accumulate_gradients(&mut state, gradients)?;

            // Handle invalid gradients found during accumulation
            if acc_result.had_invalid_gradients {
                state.consecutive_nan_count += 1;
                state.nan_gradient_count += acc_result.invalid_param_count as u64;
                warn!(
                    "Found {} parameters with NaN/Inf during accumulation: {:?} (consecutive: {})",
                    acc_result.invalid_param_count,
                    acc_result.invalid_param_names,
                    state.consecutive_nan_count
                );

                // Check emergency save threshold
                let emergency_threshold = config.emergency_save_threshold.unwrap_or(5) as u32;
                if state.consecutive_nan_count >= emergency_threshold && !state.needs_emergency_save
                {
                    state.needs_emergency_save = true;
                    warn!(
                        "Emergency save triggered: {} consecutive steps with NaN/Inf gradients",
                        state.consecutive_nan_count
                    );
                }
            } else {
                // Reset consecutive NaN count on fully successful accumulation
                state.consecutive_nan_count = 0;
            }

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

                apply_optimizer_step(
                    &mut model_mut,
                    &grads,
                    &params,
                    lr,
                    &optimizer_arc,
                    grad_acc_steps,
                )?;
                drop(model_mut);
                drop(grads);

                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.accumulated_gradients = None;
                state.micro_step = 0;
                state.step += 1;
                state.epoch_steps += 1;
                drop(state);

                heavy_cleanup();
                true
            } else {
                state.step += 1;
                state.epoch_steps += 1;
                drop(state);
                heavy_cleanup();
                false
            };

            info!(
                "Gradients applied in {:.1}s (applied={})",
                apply_start.elapsed().as_secs_f64(),
                gradients_applied
            );

            let (mean_reward, std_reward) = compute_reward_stats(&rewards_clone);
            let total_tokens: u32 = filtered_token_counts.iter().sum();

            {
                let mut state = state_arc.write().map_err(|_| {
                    Error::new(Status::GenericFailure, "Failed to acquire state write lock")
                })?;
                state.epoch_loss_sum += loss_value;
                state.epoch_reward_sum += mean_reward;
                state.epoch_tokens += total_tokens as i64;
            }

            let rewards_f32: Vec<f32> = rewards_clone.iter().map(|&r| r as f32).collect();
            let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards_clone.len() as i64])?;
            let advantages =
                compute_advantages(&rewards_array, group_size_for_training, "group".to_string())?;
            let adv_data = advantages.to_float32()?;
            let mean_advantage =
                adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len() as f64;
            let std_advantage = {
                let variance = adv_data
                    .iter()
                    .map(|&a| {
                        let diff = a as f64 - mean_advantage;
                        diff * diff
                    })
                    .sum::<f64>()
                    / adv_data.len() as f64;
                variance.sqrt()
            };

            // Note: heavy_cleanup() was already called after gradient application above.
            // No need for additional cleanup here - the compiled graph cache was already cleared.

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
                std_advantage,
                total_tokens: total_tokens as i32,
                gradients_applied,
                generation_time_ms,
                training_time_ms: training_start.elapsed().as_secs_f64() * 1000.0,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
            })
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("spawn_blocking error in training: {}", e),
            )
        })??;

        Ok(TrainStepResultWithOutputs {
            metrics,
            completions: completion_texts,
            rewards,
            outputs_json: outputs_json_for_return,
            completion_lengths: token_counts.iter().map(|&x| x as i32).collect(),
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

        // Reset optimizer state if present
        if let Some(ref optimizer) = self.optimizer
            && let Ok(mut opt) = optimizer.lock()
        {
            opt.reset();
        }

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

    /// Save optimizer state (moment tensors + step) to a SafeTensors file.
    ///
    /// The step counter is stored in the `__metadata__` field.
    /// Each parameter's first moment (m) and second moment (v) are stored as
    /// `{param_name}.m` and `{param_name}.v` tensors.
    ///
    /// No-op if the engine uses SGD (no optimizer state to save).
    #[napi]
    pub fn save_optimizer_state(&self, path: String) -> Result<()> {
        let opt_arc = match &self.optimizer {
            Some(opt) => opt,
            None => return Ok(()), // SGD — no state to save
        };

        let opt = opt_arc
            .lock()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire optimizer lock"))?;

        let step = opt.get_step();
        let keys = opt.get_state_keys();

        if keys.is_empty() {
            return Ok(()); // No state accumulated yet
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

        crate::utils::safetensors::save_safetensors(&path, &tensors, Some(metadata))
    }

    /// Load optimizer state (moment tensors + step) from a SafeTensors file.
    ///
    /// Restores the step counter from metadata and sets first/second moment
    /// tensors for each parameter found in the file.
    ///
    /// No-op if the engine uses SGD (no optimizer to restore).
    #[napi]
    pub fn load_optimizer_state(&self, path: String) -> Result<()> {
        let opt_arc = match &self.optimizer {
            Some(opt) => opt,
            None => return Ok(()), // SGD — nothing to restore
        };

        let mut opt = opt_arc
            .lock()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire optimizer lock"))?;

        // Load SafeTensors file
        let st_file = crate::utils::safetensors::SafeTensorsFile::load(&path)?;

        // Restore step from metadata
        if let Some(metadata) = &st_file.metadata
            && let Some(step_str) = metadata.get("step").and_then(|v| v.as_str())
            && let Ok(step) = step_str.parse::<i64>()
        {
            opt.set_step(step);
        }

        // Load all tensors
        let tensors = st_file.load_tensors(&path)?;

        // Restore moment tensors: keys are "{param_name}.m" and "{param_name}.v"
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
    /// Finish reasons for each completion ("stop", "length", or "repetition")
    finish_reasons: Vec<String>,
}

// =============================================================================
// Helper functions
// =============================================================================

/// Result of gradient accumulation
struct AccumulationResult {
    /// Whether any gradients contained NaN/Inf values
    had_invalid_gradients: bool,
    /// Number of parameters that had invalid gradients
    invalid_param_count: usize,
    /// Names of parameters with invalid gradients (for logging)
    invalid_param_names: Vec<String>,
}

/// Accumulate gradients into state with finite value checking.
///
/// Gradients are checked for NaN/Inf values before accumulating. If a gradient
/// contains non-finite values, it is skipped to prevent corrupting the accumulated
/// gradients. The function returns information about any skipped gradients.
///
/// We eval() each accumulated gradient to materialize it and allow MLX to free
/// the computation graph.
fn accumulate_gradients(
    state: &mut EngineState,
    new_grads: HashMap<String, MxArray>,
) -> Result<AccumulationResult> {
    let mut invalid_param_names = Vec::new();

    match &mut state.accumulated_gradients {
        Some(acc) => {
            for (name, grad) in new_grads {
                // Check for non-finite values before accumulating
                // This uses GPU-native isfinite() which is efficient
                grad.eval();
                if grad.has_nan_or_inf()? {
                    warn!(
                        "Skipping gradient accumulation for '{}' due to NaN/Inf values",
                        name
                    );
                    invalid_param_names.push(name);
                    // Skip this gradient, keep existing accumulated value
                    continue;
                }

                if let Some(existing) = acc.get_mut(&name) {
                    let summed = existing.add(&grad)?;
                    // CRITICAL: eval() to materialize the result, allowing MLX to free
                    // the computation graph from the add operation
                    summed.eval();
                    *existing = summed;
                } else {
                    // First accumulation for this parameter - just store
                    acc.insert(name, grad);
                }
            }
        }
        None => {
            // First step - filter out invalid gradients and eval valid ones
            let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
            for (name, grad) in new_grads {
                grad.eval();
                if grad.has_nan_or_inf()? {
                    warn!(
                        "Skipping initial gradient for '{}' due to NaN/Inf values",
                        name
                    );
                    invalid_param_names.push(name);
                    continue;
                }
                evaluated_grads.insert(name, grad);
            }
            state.accumulated_gradients = Some(evaluated_grads);
        }
    }

    let invalid_param_count = invalid_param_names.len();
    Ok(AccumulationResult {
        had_invalid_gradients: invalid_param_count > 0,
        invalid_param_count,
        invalid_param_names,
    })
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

    #[test]
    fn test_accumulate_gradients_valid() {
        let mut state = EngineState::default();
        let mut grads = HashMap::new();
        grads.insert(
            "param1".to_string(),
            MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        grads.insert(
            "param2".to_string(),
            MxArray::from_float32(&[4.0, 5.0], &[2]).unwrap(),
        );

        let result = accumulate_gradients(&mut state, grads).unwrap();
        assert!(!result.had_invalid_gradients);
        assert_eq!(result.invalid_param_count, 0);
        assert!(result.invalid_param_names.is_empty());
        assert!(state.accumulated_gradients.is_some());
        assert_eq!(state.accumulated_gradients.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_accumulate_gradients_with_nan() {
        let mut state = EngineState::default();
        let mut grads = HashMap::new();
        grads.insert(
            "valid".to_string(),
            MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        grads.insert(
            "invalid_nan".to_string(),
            MxArray::from_float32(&[1.0, f32::NAN, 3.0], &[3]).unwrap(),
        );

        let result = accumulate_gradients(&mut state, grads).unwrap();
        assert!(result.had_invalid_gradients);
        assert_eq!(result.invalid_param_count, 1);
        assert!(
            result
                .invalid_param_names
                .contains(&"invalid_nan".to_string())
        );

        // Valid gradient should still be accumulated
        assert!(state.accumulated_gradients.is_some());
        let acc = state.accumulated_gradients.as_ref().unwrap();
        assert_eq!(acc.len(), 1);
        assert!(acc.contains_key("valid"));
    }

    #[test]
    fn test_accumulate_gradients_with_inf() {
        let mut state = EngineState::default();
        let mut grads = HashMap::new();
        grads.insert(
            "valid".to_string(),
            MxArray::from_float32(&[1.0, 2.0], &[2]).unwrap(),
        );
        grads.insert(
            "invalid_inf".to_string(),
            MxArray::from_float32(&[f32::INFINITY, 2.0], &[2]).unwrap(),
        );
        grads.insert(
            "invalid_neg_inf".to_string(),
            MxArray::from_float32(&[1.0, f32::NEG_INFINITY], &[2]).unwrap(),
        );

        let result = accumulate_gradients(&mut state, grads).unwrap();
        assert!(result.had_invalid_gradients);
        assert_eq!(result.invalid_param_count, 2);

        // Valid gradient should still be accumulated
        assert!(state.accumulated_gradients.is_some());
        let acc = state.accumulated_gradients.as_ref().unwrap();
        assert_eq!(acc.len(), 1);
        assert!(acc.contains_key("valid"));
    }

    #[test]
    fn test_accumulate_gradients_multiple_steps() {
        let mut state = EngineState::default();

        // First step: valid gradients
        let mut grads1 = HashMap::new();
        grads1.insert(
            "param".to_string(),
            MxArray::from_float32(&[1.0, 1.0, 1.0], &[3]).unwrap(),
        );
        let result1 = accumulate_gradients(&mut state, grads1).unwrap();
        assert!(!result1.had_invalid_gradients);

        // Second step: also valid
        let mut grads2 = HashMap::new();
        grads2.insert(
            "param".to_string(),
            MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap(),
        );
        let result2 = accumulate_gradients(&mut state, grads2).unwrap();
        assert!(!result2.had_invalid_gradients);

        // Check accumulated values (should be sum: [3, 3, 3])
        let acc = state.accumulated_gradients.as_ref().unwrap();
        let values = acc.get("param").unwrap().to_float32().unwrap();
        assert_eq!(values.as_ref(), &[3.0f32, 3.0, 3.0]);
    }

    #[test]
    fn test_accumulate_gradients_skips_nan_preserves_existing() {
        let mut state = EngineState::default();

        // First step: valid gradient
        let mut grads1 = HashMap::new();
        grads1.insert(
            "param".to_string(),
            MxArray::from_float32(&[1.0, 1.0, 1.0], &[3]).unwrap(),
        );
        let _ = accumulate_gradients(&mut state, grads1).unwrap();

        // Second step: NaN gradient (should be skipped, preserving [1, 1, 1])
        let mut grads2 = HashMap::new();
        grads2.insert(
            "param".to_string(),
            MxArray::from_float32(&[f32::NAN, 2.0, 2.0], &[3]).unwrap(),
        );
        let result2 = accumulate_gradients(&mut state, grads2).unwrap();
        assert!(result2.had_invalid_gradients);

        // Accumulated values should still be [1, 1, 1] (NaN gradient was skipped)
        let acc = state.accumulated_gradients.as_ref().unwrap();
        let values = acc.get("param").unwrap().to_float32().unwrap();
        assert_eq!(values.as_ref(), &[1.0f32, 1.0, 1.0]);
    }
}
