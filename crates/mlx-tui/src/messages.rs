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
    Step {
        step: u64,
        loss: f64,
        #[serde(rename = "meanReward")]
        mean_reward: f64,
        #[serde(rename = "stdReward")]
        std_reward: f64,
        #[serde(rename = "meanAdvantage")]
        mean_advantage: f64,
        #[serde(rename = "totalTokens")]
        total_tokens: u32,
        #[serde(rename = "generationTimeMs")]
        generation_time_ms: f64,
        #[serde(rename = "trainingTimeMs")]
        training_time_ms: f64,
    },

    /// Generated completion sample
    Generation {
        index: u32,
        prompt: String,
        completion: String,
        reward: f64,
        tokens: u32,
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
    #[serde(rename = "maxNewTokens")]
    pub max_new_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(rename = "clipEpsilon")]
    pub clip_epsilon: Option<f64>,
    #[serde(rename = "gradientAccumulationSteps")]
    pub gradient_accumulation_steps: Option<u32>,
    #[serde(rename = "lossType")]
    pub loss_type: Option<String>,

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
