//! Data types for the output store
//!
//! Plain Rust types for database records (no NAPI decorators).
//! These types are shared between mlx-core (NAPI) and mlx-tui (native).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for creating a database connection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DbConfig {
    /// Local SQLite file path (e.g., "training_outputs.db")
    pub local_path: String,
}

/// A training run record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingRunRecord {
    /// Unique run ID (UUID)
    pub id: String,
    /// Human-readable run name (for resume)
    pub name: Option<String>,
    /// Model name
    pub model_name: String,
    /// Path to model weights
    pub model_path: Option<String>,
    /// Serialized training config (JSON)
    pub config: String,
    /// Unix timestamp (milliseconds) when training started
    pub started_at: i64,
    /// Unix timestamp (milliseconds) when training ended
    pub ended_at: Option<i64>,
    /// Total number of training steps completed
    pub total_steps: i64,
    /// Run status: "running", "completed", "failed", "paused"
    pub status: String,
}

/// Database row for training runs (for sqlx FromRow)
#[derive(sqlx::FromRow)]
pub struct TrainingRunRow {
    pub id: String,
    pub name: Option<String>,
    pub model_name: String,
    pub model_path: Option<String>,
    pub config: String,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub total_steps: i64,
    pub status: String,
}

impl From<TrainingRunRow> for TrainingRunRecord {
    fn from(row: TrainingRunRow) -> Self {
        Self {
            id: row.id,
            name: row.name,
            model_name: row.model_name,
            model_path: row.model_path,
            config: row.config,
            started_at: row.started_at,
            ended_at: row.ended_at,
            total_steps: row.total_steps,
            status: row.status,
        }
    }
}

/// A training step record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepRecord {
    /// Run ID this step belongs to
    pub run_id: String,
    /// Step number
    pub step: i64,
    /// Epoch number
    pub epoch: Option<i64>,
    /// GRPO loss value
    pub loss: f64,
    /// Mean reward across completions
    pub mean_reward: f64,
    /// Standard deviation of rewards
    pub std_reward: f64,
    /// Mean advantage value
    pub mean_advantage: Option<f64>,
    /// Std advantage value - indicates reward variance within groups
    pub std_advantage: f64,
    /// Total tokens generated this step
    pub total_tokens: Option<i64>,
    /// Time for generation phase (milliseconds)
    pub generation_time_ms: Option<f64>,
    /// Time for training phase (milliseconds)
    pub training_time_ms: Option<f64>,
    /// Whether gradients were applied this step
    pub gradients_applied: bool,
}

/// A generation record (one completion)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationRecord {
    /// Database ID (for referencing in queries)
    pub id: Option<i64>,
    /// Step number this generation belongs to
    pub step: Option<i64>,
    /// Index within the batch
    pub batch_index: i64,
    /// Index within the group (0 to group_size-1)
    pub group_index: i64,
    /// The prompt text
    pub prompt: String,
    /// Expected answer (if available)
    pub expected_answer: Option<String>,
    /// Cleaned completion text (tags removed)
    pub completion_text: String,
    /// Raw completion text (with <think>/<tool_call> tags)
    pub completion_raw: String,
    /// Extracted thinking content from <think> tags
    pub thinking: Option<String>,
    /// Number of tokens in the completion
    pub num_tokens: i64,
    /// Finish reason: "eos", "length", or "repetition"
    pub finish_reason: String,
    /// Reward value for this completion
    pub reward: f64,
}

/// Database row for generations (for sqlx FromRow)
#[derive(sqlx::FromRow)]
pub struct GenerationRow {
    pub id: i64,
    pub step: i64,
    pub batch_index: i64,
    pub group_index: i64,
    pub prompt: String,
    pub expected_answer: Option<String>,
    pub completion_text: String,
    pub completion_raw: String,
    pub thinking: Option<String>,
    pub num_tokens: i64,
    pub finish_reason: String,
    pub reward: f64,
}

impl From<GenerationRow> for GenerationRecord {
    fn from(row: GenerationRow) -> Self {
        Self {
            id: Some(row.id),
            step: Some(row.step),
            batch_index: row.batch_index,
            group_index: row.group_index,
            prompt: row.prompt,
            expected_answer: row.expected_answer,
            completion_text: row.completion_text,
            completion_raw: row.completion_raw,
            thinking: row.thinking,
            num_tokens: row.num_tokens,
            finish_reason: row.finish_reason,
            reward: row.reward,
        }
    }
}

/// A tool call record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Database ID
    pub id: Option<i64>,
    /// Generation ID this tool call belongs to
    pub generation_id: Option<i64>,
    /// Index of this call within the generation
    pub call_index: i64,
    /// Parse status: "ok", "parse_error", "json_error"
    pub status: String,
    /// Tool name (null if parse failed)
    pub tool_name: Option<String>,
    /// Tool arguments as JSON (null if parse failed)
    pub arguments: Option<String>,
    /// Raw content from <tool_call> tag
    pub raw_content: String,
    /// Error message if parsing failed
    pub error_message: Option<String>,
}

/// Database row for tool calls (for sqlx FromRow)
#[derive(sqlx::FromRow)]
pub struct ToolCallRow {
    pub id: i64,
    pub call_index: i64,
    pub status: String,
    pub tool_name: Option<String>,
    pub arguments: Option<String>,
    pub raw_content: String,
    pub error_message: Option<String>,
}

impl ToolCallRow {
    pub fn into_record(self, generation_id: i64) -> ToolCallRecord {
        ToolCallRecord {
            id: Some(self.id),
            generation_id: Some(generation_id),
            call_index: self.call_index,
            status: self.status,
            tool_name: self.tool_name,
            arguments: self.arguments,
            raw_content: self.raw_content,
            error_message: self.error_message,
        }
    }
}

/// A generation with its associated tool calls
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationWithToolCalls {
    /// The generation record
    pub generation: GenerationRecord,
    /// Tool calls made in this generation
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Summary of a training step
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepSummary {
    /// Step number
    pub step: i64,
    /// Loss value
    pub loss: f64,
    /// Mean reward
    pub mean_reward: f64,
    /// Mean advantage
    pub mean_advantage: f64,
    /// Standard deviation of advantages - indicates reward variance within groups
    pub std_advantage: f64,
    /// Number of generations in this step
    pub num_generations: i64,
    /// Number of tool calls across all generations
    pub num_tool_calls: i64,
    /// Count of completions that ended with EOS
    pub eos_count: i64,
    /// Count of completions that hit token limit
    pub length_count: i64,
}

/// Database row for step summaries (for sqlx FromRow)
#[derive(sqlx::FromRow)]
pub struct StepSummaryRow {
    pub step: i64,
    pub loss: f64,
    pub mean_reward: f64,
    pub mean_advantage: f64,
    pub std_advantage: f64,
    pub num_generations: i64,
    pub num_tool_calls: i64,
    pub eos_count: i64,
    pub length_count: i64,
}

impl From<StepSummaryRow> for StepSummary {
    fn from(row: StepSummaryRow) -> Self {
        Self {
            step: row.step,
            loss: row.loss,
            mean_reward: row.mean_reward,
            mean_advantage: row.mean_advantage,
            std_advantage: row.std_advantage,
            num_generations: row.num_generations,
            num_tool_calls: row.num_tool_calls,
            eos_count: row.eos_count,
            length_count: row.length_count,
        }
    }
}

/// Reward distribution statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewardStats {
    /// Total count of generations
    pub count: i64,
    /// Mean reward
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum reward
    pub min: f64,
    /// Maximum reward
    pub max: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// 25th percentile
    pub p25: f64,
    /// 75th percentile
    pub p75: f64,
}

impl Default for RewardStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p25: 0.0,
            p75: 0.0,
        }
    }
}

/// Filter options for querying generations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerationFilter {
    /// Minimum step number
    pub step_min: Option<i64>,
    /// Maximum step number
    pub step_max: Option<i64>,
    /// Minimum reward value
    pub reward_min: Option<f64>,
    /// Maximum reward value
    pub reward_max: Option<f64>,
    /// Filter by finish reason ("eos", "length", "repetition")
    pub finish_reason: Option<String>,
    /// Filter to only generations with tool calls
    pub has_tool_calls: Option<bool>,
    /// Filter by tool call status ("ok", "parse_error", "json_error")
    pub tool_call_status: Option<String>,
    /// Maximum number of results
    pub limit: Option<i64>,
    /// Offset for pagination
    pub offset: Option<i64>,
}

/// A log record for persisting application logs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogRecord {
    /// Database ID
    pub id: Option<i64>,
    /// Run ID this log belongs to (nullable for pre-init logs)
    pub run_id: Option<String>,
    /// Log level: DEBUG, INFO, WARN, ERROR
    pub level: String,
    /// Module/target path (e.g., mlx_tui::app)
    pub target: String,
    /// Log message content
    pub message: String,
    /// Source file path
    pub file: Option<String>,
    /// Source line number
    pub line: Option<u32>,
    /// Unix timestamp (milliseconds)
    pub created_at: i64,
}

/// Database row for logs (for sqlx FromRow)
#[derive(sqlx::FromRow)]
pub struct LogRow {
    pub id: i64,
    pub run_id: Option<String>,
    pub level: String,
    pub target: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<i64>,
    pub created_at: i64,
}

impl From<LogRow> for LogRecord {
    fn from(row: LogRow) -> Self {
        Self {
            id: Some(row.id),
            run_id: row.run_id,
            level: row.level,
            target: row.target,
            message: row.message,
            file: row.file,
            line: row.line.map(|l| l as u32),
            created_at: row.created_at,
        }
    }
}

/// Metrics from a single training step
///
/// Shared between mlx-core (NAPI) and mlx-db for training output persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

// =============================================================================
// RewardOutput types (local copy to avoid mlx-core dependency)
// These mirror the types from mlx-core/src/tools/mod.rs for JSON deserialization
// =============================================================================

/// Structured tool call result (for deserialization from RewardOutput JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Unique identifier for this tool call
    pub id: String,
    /// Name of the tool/function to call
    pub name: String,
    /// Parsed arguments as JSON value
    pub arguments: Value,
    /// Parsing status: "ok" | "invalid_json" | "missing_name"
    pub status: String,
    /// Error message if status != "ok"
    pub error: Option<String>,
    /// Raw content from <tool_call> tag
    #[serde(default)]
    pub raw_content: String,
}

/// Structured completion information (for deserialization from RewardOutput JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionInfo {
    /// Clean text with <tool_call> and <think> tags removed
    pub text: String,
    /// Raw output before tag stripping
    pub raw_text: String,
    /// Parsed tool calls
    pub tool_calls: Vec<ToolCallResult>,
    /// Extracted thinking/reasoning from <think> tags
    pub thinking: Option<String>,
    /// Number of tokens generated
    pub num_tokens: u32,
    /// Finish reason: "stop" | "length" | "tool_calls"
    pub finish_reason: String,
}

/// Reward function input for a single completion (for deserialization from RewardOutput JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardOutput {
    /// The input prompt text
    pub prompt: String,
    /// Structured completion data
    pub completion: CompletionInfo,
}

// =============================================================================
// Types for TUI resume state restoration
// =============================================================================

/// Metrics from a single training step (for sparkline restoration)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMetricSummary {
    /// Step number
    pub step: i64,
    /// Loss value
    pub loss: f64,
    /// Mean reward (GRPO)
    pub mean_reward: f64,
    /// Mean advantage (GRPO)
    pub mean_advantage: f64,
    /// Std advantage (GRPO) - indicates reward variance within groups
    pub std_advantage: f64,
    /// Perplexity (SFT, optional)
    pub perplexity: Option<f64>,
    /// Token accuracy (SFT, optional)
    pub token_accuracy: Option<f64>,
    /// Total tokens this step
    pub total_tokens: i64,
    /// Time for generation phase (milliseconds)
    pub generation_time_ms: Option<f64>,
    /// Time for training phase (milliseconds)
    pub training_time_ms: Option<f64>,
}

/// Database row for step metric summaries
#[derive(sqlx::FromRow)]
pub struct StepMetricRow {
    pub step: i64,
    pub loss: f64,
    pub mean_reward: f64,
    pub mean_advantage: f64,
    pub std_advantage: f64,
    pub total_tokens: Option<i64>,
    pub generation_time_ms: Option<f64>,
    pub training_time_ms: Option<f64>,
}

impl From<StepMetricRow> for StepMetricSummary {
    fn from(row: StepMetricRow) -> Self {
        Self {
            step: row.step,
            loss: row.loss,
            mean_reward: row.mean_reward,
            mean_advantage: row.mean_advantage,
            std_advantage: row.std_advantage,
            perplexity: None, // SFT metrics not stored in DB yet
            token_accuracy: None,
            total_tokens: row.total_tokens.unwrap_or(0),
            generation_time_ms: row.generation_time_ms,
            training_time_ms: row.training_time_ms,
        }
    }
}

/// Aggregate statistics for a training run (for resume state)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunAggregates {
    /// Best (highest) reward seen
    pub best_reward: f64,
    /// Average reward
    pub avg_reward: f64,
    /// Total reward count (for incremental average)
    pub reward_count: i64,
    /// Best (lowest) loss seen
    pub best_loss: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Total loss count (for incremental average)
    pub loss_count: i64,
    /// Total tokens generated across all steps
    pub total_tokens: i64,
    /// Current step number
    pub current_step: i64,
    /// Average generation time (milliseconds)
    pub avg_generation_time_ms: f64,
    /// Average training time (milliseconds)
    pub avg_training_time_ms: f64,
}

/// Database row for run aggregates query
#[derive(sqlx::FromRow)]
pub struct RunAggregatesRow {
    pub best_reward: f64,
    pub avg_reward: f64,
    pub reward_count: i64,
    pub best_loss: f64,
    pub avg_loss: f64,
    pub loss_count: i64,
    pub total_tokens: i64,
    pub current_step: i64,
    pub avg_generation_time_ms: f64,
    pub avg_training_time_ms: f64,
}

impl From<RunAggregatesRow> for RunAggregates {
    fn from(row: RunAggregatesRow) -> Self {
        Self {
            best_reward: row.best_reward,
            avg_reward: row.avg_reward,
            reward_count: row.reward_count,
            best_loss: row.best_loss,
            avg_loss: row.avg_loss,
            loss_count: row.loss_count,
            total_tokens: row.total_tokens,
            current_step: row.current_step,
            avg_generation_time_ms: row.avg_generation_time_ms,
            avg_training_time_ms: row.avg_training_time_ms,
        }
    }
}
