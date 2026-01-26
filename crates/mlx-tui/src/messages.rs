//! JSONL message types from training process
//!
//! These types represent the structured messages sent from the Node.js
//! training process to the TUI via stdout.

use serde::Deserialize;
use std::collections::HashMap;

/// All possible messages from the training process
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainingMessage {
    /// Training initialization with model info and config
    Init {
        model: String,
        config: TrainingConfig,
    },

    /// Start of a new epoch
    EpochStart {
        epoch: u32,
        #[serde(rename = "totalEpochs")]
        total_epochs: u32,
        #[serde(rename = "numBatches")]
        num_batches: u32,
    },

    /// Training step completed
    ///
    /// Supports both SFT and GRPO metrics:
    /// - SFT: loss, perplexity, token_accuracy (no reward/advantage)
    /// - GRPO: loss, mean_reward, std_reward, mean_advantage
    Step {
        step: u64,
        loss: f64,
        /// GRPO-specific: mean reward (optional for SFT)
        #[serde(rename = "meanReward")]
        mean_reward: Option<f64>,
        /// GRPO-specific: reward std dev (optional for SFT)
        #[serde(rename = "stdReward")]
        std_reward: Option<f64>,
        /// GRPO-specific: std advantage - indicates reward variance within groups
        #[serde(rename = "stdAdvantage")]
        std_advantage: Option<f64>,
        /// SFT-specific: perplexity = exp(loss)
        #[serde(rename = "perplexity")]
        perplexity: Option<f64>,
        /// SFT-specific: token-level accuracy
        #[serde(rename = "tokenAccuracy")]
        token_accuracy: Option<f64>,
        #[serde(rename = "totalTokens")]
        total_tokens: u32,
        #[serde(rename = "generationTimeMs")]
        generation_time_ms: Option<f64>,
        #[serde(rename = "trainingTimeMs")]
        training_time_ms: Option<f64>,
        #[serde(rename = "peakMemoryMb")]
        peak_memory_mb: Option<f64>,
        #[serde(rename = "activeMemoryMb")]
        active_memory_mb: Option<f64>,
    },

    /// Generated completion sample
    Generation {
        index: u32,
        prompt: String,
        completion: String,
        reward: f64,
        tokens: u32,
        #[serde(rename = "rewardDetails")]
        reward_details: Option<std::collections::HashMap<String, f64>>,
    },

    /// Checkpoint saved
    Checkpoint { path: String, step: u64 },

    /// Epoch completed
    EpochEnd {
        epoch: u32,
        #[serde(rename = "avgLoss")]
        avg_loss: f64,
        #[serde(rename = "avgReward")]
        avg_reward: f64,
        #[serde(rename = "epochTimeSecs")]
        epoch_time_secs: f64,
    },

    /// Training completed
    Complete {
        #[serde(rename = "totalSteps")]
        total_steps: u64,
        #[serde(rename = "totalTimeSecs")]
        total_time_secs: f64,
    },

    /// Log message
    Log { level: LogLevel, message: String },

    /// Training paused
    Paused { step: u64 },

    /// Training resumed
    Resumed { step: u64 },

    /// Status update during initialization (for loading progress)
    Status { phase: String, message: String },

    /// Database path from training process (for DB tab)
    DatabasePath {
        path: String,
        #[serde(rename = "runId")]
        run_id: String,
        #[serde(rename = "runName")]
        run_name: Option<String>,
    },

    /// Interactive prompt from training process
    /// Training script sends this to ask user for input
    Prompt {
        /// Unique ID to identify the prompt response
        id: String,
        /// Message to display
        message: String,
        /// Available choices (value, display label)
        choices: Vec<PromptChoice>,
        /// Optional default selection index (single-select) or indices (multi-select)
        #[serde(default)]
        default: Option<Vec<usize>>,
        /// Allow multiple selections (default: false)
        #[serde(default, rename = "multiSelect")]
        multi_select: bool,
    },

    /// Resume state from training script (sent when resuming from checkpoint)
    ///
    /// Contains historical metrics for sparkline restoration and aggregate statistics.
    /// The training script queries the database and sends this message to restore
    /// TUI state when resuming a training run.
    ResumeState {
        /// Current step number
        step: u64,
        /// Current epoch number
        epoch: u32,
        /// Total epochs for progress display
        #[serde(rename = "totalEpochs")]
        total_epochs: u32,
        /// Current batch within epoch (for progress display)
        #[serde(rename = "stepInEpoch", default)]
        step_in_epoch: u32,
        /// Total batches per epoch (for progress display)
        #[serde(rename = "totalStepsInEpoch", default)]
        total_steps_in_epoch: u32,
        /// Historical metrics for sparklines (oldest first)
        #[serde(rename = "metricsHistory")]
        metrics_history: Vec<ResumeMetric>,
        /// Aggregate statistics for the run
        aggregates: ResumeAggregates,
    },
}

/// Historical metric for sparkline restoration
#[derive(Debug, Clone, Deserialize)]
pub struct ResumeMetric {
    /// Step number (included for JSON compatibility, not used in TUI)
    #[allow(dead_code)]
    pub step: i64,
    /// Loss value
    pub loss: f64,
    /// Mean reward (GRPO)
    #[serde(rename = "meanReward")]
    pub mean_reward: f64,
    /// Std advantage (GRPO) - indicates reward variance within groups
    #[serde(rename = "stdAdvantage", default)]
    pub std_advantage: f64,
    /// Perplexity (SFT, optional)
    pub perplexity: Option<f64>,
    /// Token accuracy (SFT, optional)
    #[serde(rename = "tokenAccuracy")]
    pub token_accuracy: Option<f64>,
    /// Time for generation phase (milliseconds)
    #[serde(rename = "generationTimeMs")]
    pub generation_time_ms: Option<f64>,
    /// Time for training phase (milliseconds)
    #[serde(rename = "trainingTimeMs")]
    pub training_time_ms: Option<f64>,
}

/// Aggregate statistics for resume state
#[derive(Debug, Clone, Deserialize)]
pub struct ResumeAggregates {
    /// Best (highest) reward seen
    #[serde(rename = "bestReward")]
    pub best_reward: f64,
    /// Average reward
    #[serde(rename = "avgReward")]
    pub avg_reward: f64,
    /// Total reward count
    #[serde(rename = "rewardCount")]
    pub reward_count: i64,
    /// Best (lowest) loss seen
    #[serde(rename = "bestLoss")]
    pub best_loss: f64,
    /// Average loss
    #[serde(rename = "avgLoss")]
    pub avg_loss: f64,
    /// Total loss count
    #[serde(rename = "lossCount")]
    pub loss_count: i64,
    /// Total tokens generated
    #[serde(rename = "totalTokens")]
    pub total_tokens: i64,
    /// Average generation time (milliseconds)
    #[serde(rename = "avgGenerationTimeMs", default)]
    pub avg_generation_time_ms: f64,
    /// Average training time (milliseconds)
    #[serde(rename = "avgTrainingTimeMs", default)]
    pub avg_training_time_ms: f64,
}

/// A choice in an interactive prompt
#[derive(Debug, Clone, Deserialize)]
pub struct PromptChoice {
    /// Value to send back when selected
    pub value: String,
    /// Display label shown to user
    pub label: String,
    /// Optional description/hint
    pub description: Option<String>,
}

/// Training configuration from init message
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TrainingConfig {
    #[serde(rename = "learningRate")]
    pub learning_rate: Option<f64>,
    #[serde(rename = "batchSize")]
    pub batch_size: Option<u32>,
    #[serde(rename = "groupSize")]
    pub group_size: Option<u32>,
    #[serde(rename = "numEpochs")]
    pub num_epochs: Option<u32>,
    #[serde(rename = "maxCompletionLength")]
    pub max_completion_length: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(rename = "clipEpsilon")]
    pub clip_epsilon: Option<f64>,
    #[serde(rename = "gradientAccumulationSteps")]
    pub gradient_accumulation_steps: Option<u32>,
    #[serde(rename = "lossType")]
    pub loss_type: Option<String>,
    /// Training type: "sft" or "grpo" (defaults to "grpo" if not specified)
    #[serde(rename = "trainingType")]
    pub training_type: Option<String>,

    /// Catch-all for unknown fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Log level for log messages
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// Cycle to the next filter level
    pub fn next_filter(self) -> Self {
        match self {
            Self::Debug => Self::Info,
            Self::Info => Self::Warn,
            Self::Warn => Self::Error,
            Self::Error => Self::Debug,
        }
    }

    /// Get display name for filter UI
    pub fn filter_name(&self) -> &'static str {
        match self {
            Self::Debug => "Debug+",
            Self::Info => "Info+",
            Self::Warn => "Warn+",
            Self::Error => "Error",
        }
    }
}

impl LogLevel {
    /// Get the color for this log level
    pub fn color(&self) -> ratatui::style::Color {
        match self {
            LogLevel::Info => ratatui::style::Color::White,
            LogLevel::Warn => ratatui::style::Color::Yellow,
            LogLevel::Error => ratatui::style::Color::Red,
            LogLevel::Debug => ratatui::style::Color::Gray,
        }
    }

    /// Get the display prefix for this log level
    pub fn prefix(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERR ",
            LogLevel::Debug => "DBG ",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_init_message() {
        let json = r#"{"type":"init","model":"qwen3-0.6b-instruct","config":{"trainingType":"grpo","numEpochs":10,"batchSize":1,"groupSize":4,"learningRate":0.000001}}"#;
        let result: Result<TrainingMessage, _> = serde_json::from_str(json);
        match &result {
            Ok(msg) => println!("Parsed: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }
        assert!(
            result.is_ok(),
            "Failed to parse init message: {:?}",
            result.err()
        );

        if let Ok(TrainingMessage::Init { model, config }) = result {
            assert_eq!(model, "qwen3-0.6b-instruct");
            assert_eq!(config.training_type, Some("grpo".to_string()));
            assert_eq!(config.num_epochs, Some(10));
            assert_eq!(config.batch_size, Some(1));
            assert_eq!(config.group_size, Some(4));
            assert_eq!(config.learning_rate, Some(0.000001));
        }
    }

    #[test]
    fn test_parse_epoch_start_message() {
        let json = r#"{"type":"epoch_start","epoch":1,"totalEpochs":10,"numBatches":100}"#;
        let result: Result<TrainingMessage, _> = serde_json::from_str(json);
        match &result {
            Ok(msg) => println!("Parsed: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }
        assert!(
            result.is_ok(),
            "Failed to parse epoch_start message: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_step_message() {
        // Test GRPO step message
        let json = r#"{"type":"step","step":1,"loss":0.5,"totalTokens":100,"meanReward":0.8,"stdReward":0.1,"meanAdvantage":0.2}"#;
        let result: Result<TrainingMessage, _> = serde_json::from_str(json);
        match &result {
            Ok(msg) => println!("Parsed: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }
        assert!(
            result.is_ok(),
            "Failed to parse step message: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_step_message_with_memory() {
        // Test step message with new memory fields
        let json = r#"{"type":"step","step":1,"loss":0.5,"totalTokens":100,"peakMemoryMb":1024.5,"activeMemoryMb":512.3}"#;
        let result: Result<TrainingMessage, _> = serde_json::from_str(json);
        match &result {
            Ok(msg) => println!("Parsed: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }
        assert!(
            result.is_ok(),
            "Failed to parse step message with memory: {:?}",
            result.err()
        );

        if let Ok(TrainingMessage::Step {
            peak_memory_mb,
            active_memory_mb,
            ..
        }) = result
        {
            assert_eq!(peak_memory_mb, Some(1024.5));
            assert_eq!(active_memory_mb, Some(512.3));
        }
    }
}
