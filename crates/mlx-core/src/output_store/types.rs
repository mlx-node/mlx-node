//! NAPI wrapper types for mlx_db types
//!
//! These types add `#[napi(object)]` decorators to mlx_db types for Node.js compatibility.

use napi_derive::napi;

/// A training run record (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct TrainingRunRecord {
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

impl From<mlx_db::TrainingRunRecord> for TrainingRunRecord {
    fn from(r: mlx_db::TrainingRunRecord) -> Self {
        Self {
            id: r.id,
            name: r.name,
            model_name: r.model_name,
            model_path: r.model_path,
            config: r.config,
            started_at: r.started_at,
            ended_at: r.ended_at,
            total_steps: r.total_steps,
            status: r.status,
        }
    }
}

/// A training step record (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct StepRecord {
    pub run_id: String,
    pub step: i64,
    pub epoch: Option<i64>,
    pub loss: f64,
    pub mean_reward: f64,
    pub std_reward: f64,
    pub mean_advantage: Option<f64>,
    pub std_advantage: f64,
    pub total_tokens: Option<i64>,
    pub generation_time_ms: Option<f64>,
    pub training_time_ms: Option<f64>,
    pub gradients_applied: bool,
}

impl From<mlx_db::StepRecord> for StepRecord {
    fn from(s: mlx_db::StepRecord) -> Self {
        Self {
            run_id: s.run_id,
            step: s.step,
            epoch: s.epoch,
            loss: s.loss,
            mean_reward: s.mean_reward,
            std_reward: s.std_reward,
            mean_advantage: s.mean_advantage,
            std_advantage: s.std_advantage,
            total_tokens: s.total_tokens,
            generation_time_ms: s.generation_time_ms,
            training_time_ms: s.training_time_ms,
            gradients_applied: s.gradients_applied,
        }
    }
}

/// A generation record (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct GenerationRecord {
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

impl From<mlx_db::GenerationRecord> for GenerationRecord {
    fn from(g: mlx_db::GenerationRecord) -> Self {
        Self {
            batch_index: g.batch_index,
            group_index: g.group_index,
            prompt: g.prompt,
            expected_answer: g.expected_answer,
            completion_text: g.completion_text,
            completion_raw: g.completion_raw,
            thinking: g.thinking,
            num_tokens: g.num_tokens,
            finish_reason: g.finish_reason,
            reward: g.reward,
        }
    }
}

/// A tool call record (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct ToolCallRecord {
    pub call_index: i64,
    pub status: String,
    pub tool_name: Option<String>,
    pub arguments: Option<String>,
    pub raw_content: String,
    pub error_message: Option<String>,
}

impl From<mlx_db::ToolCallRecord> for ToolCallRecord {
    fn from(tc: mlx_db::ToolCallRecord) -> Self {
        Self {
            call_index: tc.call_index,
            status: tc.status,
            tool_name: tc.tool_name,
            arguments: tc.arguments,
            raw_content: tc.raw_content,
            error_message: tc.error_message,
        }
    }
}

/// A generation with its associated tool calls (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct GenerationWithToolCalls {
    pub generation: GenerationRecord,
    pub tool_calls: Vec<ToolCallRecord>,
}

impl From<mlx_db::GenerationWithToolCalls> for GenerationWithToolCalls {
    fn from(g: mlx_db::GenerationWithToolCalls) -> Self {
        Self {
            generation: g.generation.into(),
            tool_calls: g.tool_calls.into_iter().map(Into::into).collect(),
        }
    }
}

/// Summary of a training step (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct StepSummary {
    pub step: i64,
    pub loss: f64,
    pub mean_reward: f64,
    pub num_generations: i64,
    pub num_tool_calls: i64,
    pub eos_count: i64,
    pub length_count: i64,
}

impl From<mlx_db::StepSummary> for StepSummary {
    fn from(s: mlx_db::StepSummary) -> Self {
        Self {
            step: s.step,
            loss: s.loss,
            mean_reward: s.mean_reward,
            num_generations: s.num_generations,
            num_tool_calls: s.num_tool_calls,
            eos_count: s.eos_count,
            length_count: s.length_count,
        }
    }
}

/// Reward distribution statistics (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct RewardStats {
    pub count: i64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub p25: f64,
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

impl From<mlx_db::RewardStats> for RewardStats {
    fn from(s: mlx_db::RewardStats) -> Self {
        Self {
            count: s.count,
            mean: s.mean,
            std: s.std,
            min: s.min,
            max: s.max,
            median: s.median,
            p25: s.p25,
            p75: s.p75,
        }
    }
}

// =============================================================================
// TUI Resume State Types
// =============================================================================

/// Metrics from a single training step for sparkline restoration (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
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

impl From<mlx_db::StepMetricSummary> for StepMetricSummary {
    fn from(s: mlx_db::StepMetricSummary) -> Self {
        Self {
            step: s.step,
            loss: s.loss,
            mean_reward: s.mean_reward,
            mean_advantage: s.mean_advantage,
            std_advantage: s.std_advantage,
            perplexity: s.perplexity,
            token_accuracy: s.token_accuracy,
            total_tokens: s.total_tokens,
            generation_time_ms: s.generation_time_ms,
            training_time_ms: s.training_time_ms,
        }
    }
}

/// Aggregate statistics for a training run for resume state (NAPI wrapper)
#[napi(object)]
#[derive(Clone)]
pub struct RunAggregates {
    /// Best (highest) reward seen
    pub best_reward: f64,
    /// Average reward
    pub avg_reward: f64,
    /// Total reward count
    pub reward_count: i64,
    /// Best (lowest) loss seen
    pub best_loss: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Total loss count
    pub loss_count: i64,
    /// Total tokens generated
    pub total_tokens: i64,
    /// Current step number
    pub current_step: i64,
    /// Average generation time (milliseconds)
    pub avg_generation_time_ms: f64,
    /// Average training time (milliseconds)
    pub avg_training_time_ms: f64,
}

impl Default for RunAggregates {
    fn default() -> Self {
        Self {
            best_reward: f64::NEG_INFINITY,
            avg_reward: 0.0,
            reward_count: 0,
            best_loss: f64::INFINITY,
            avg_loss: 0.0,
            loss_count: 0,
            total_tokens: 0,
            current_step: 0,
            avg_generation_time_ms: 0.0,
            avg_training_time_ms: 0.0,
        }
    }
}

impl From<mlx_db::RunAggregates> for RunAggregates {
    fn from(a: mlx_db::RunAggregates) -> Self {
        Self {
            best_reward: a.best_reward,
            avg_reward: a.avg_reward,
            reward_count: a.reward_count,
            best_loss: a.best_loss,
            avg_loss: a.avg_loss,
            loss_count: a.loss_count,
            total_tokens: a.total_tokens,
            current_step: a.current_step,
            avg_generation_time_ms: a.avg_generation_time_ms,
            avg_training_time_ms: a.avg_training_time_ms,
        }
    }
}

/// Statistics about cleanup operations (NAPI wrapper)
#[napi(object)]
#[derive(Clone, Default)]
pub struct CleanupStats {
    /// Number of training steps deleted
    pub steps_deleted: i64,
    /// Number of generations deleted
    pub generations_deleted: i64,
    /// Number of tool calls deleted
    pub tool_calls_deleted: i64,
    /// Number of logs deleted
    pub logs_deleted: i64,
}

impl From<mlx_db::CleanupStats> for CleanupStats {
    fn from(s: mlx_db::CleanupStats) -> Self {
        Self {
            steps_deleted: s.steps_deleted as i64,
            generations_deleted: s.generations_deleted as i64,
            tool_calls_deleted: s.tool_calls_deleted as i64,
            logs_deleted: s.logs_deleted as i64,
        }
    }
}
