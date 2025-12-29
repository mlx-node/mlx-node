//! Data types for the output store module
//!
//! These types are exposed via NAPI and used for recording and querying
//! training outputs.

use napi_derive::napi;

/// Configuration for creating an OutputStore connection
#[napi(object)]
#[derive(Clone, Default)]
pub struct OutputStoreConfig {
    /// Local SQLite file path (e.g., "training_outputs.db")
    pub local_path: Option<String>,
    /// Remote Turso URL (e.g., "libsql://db-name.turso.io")
    pub remote_url: Option<String>,
    /// Turso auth token
    pub auth_token: Option<String>,
    /// Sync interval in seconds for embedded replica mode (default: 60)
    pub sync_interval_secs: Option<u32>,
    /// Batch size for buffered inserts (default: 100)
    pub batch_size: Option<u32>,
}

/// A training run record
#[napi(object)]
#[derive(Clone)]
pub struct TrainingRunRecord {
    /// Unique run ID (UUID)
    pub id: String,
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

/// A training step record
#[napi(object)]
#[derive(Clone)]
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
#[napi(object)]
#[derive(Clone)]
pub struct GenerationRecord {
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

/// A tool call record
#[napi(object)]
#[derive(Clone)]
pub struct ToolCallRecord {
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

/// A generation with its associated tool calls
#[napi(object)]
#[derive(Clone)]
pub struct GenerationWithToolCalls {
    /// The generation record
    pub generation: GenerationRecord,
    /// Tool calls made in this generation
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Summary of a training step
#[napi(object)]
#[derive(Clone)]
pub struct StepSummary {
    /// Step number
    pub step: i64,
    /// Loss value
    pub loss: f64,
    /// Mean reward
    pub mean_reward: f64,
    /// Number of generations in this step
    pub num_generations: i64,
    /// Number of tool calls across all generations
    pub num_tool_calls: i64,
    /// Count of completions that ended with EOS
    pub eos_count: i64,
    /// Count of completions that hit token limit
    pub length_count: i64,
}

/// Reward distribution statistics
#[napi(object)]
#[derive(Clone)]
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
