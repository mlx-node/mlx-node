//! Application state and logic
//!
//! Contains the main App struct that holds all TUI state and handles
//! incoming messages from the training process.

use std::collections::VecDeque;

use tracing::{debug, error, info, warn};

use crate::commands::SampleDisplayMode;
use crate::messages::{
    LogLevel, PromptChoice, ResumeAggregates, ResumeMetric, TrainingConfig, TrainingMessage,
};
use mlx_db::{GenerationFilter, GenerationRecord, SyncDbReader};

/// Maximum number of data points to keep for sparklines
const SPARKLINE_HISTORY: usize = 60;
/// Maximum number of log entries to keep
const LOG_HISTORY: usize = 500;
/// Maximum number of generation samples to keep
const SAMPLE_HISTORY: usize = 50;

/// Pending database action that needs to be processed in blocking context
#[derive(Debug, Clone)]
pub enum DbAction {
    /// Refresh generations list with current filter
    RefreshGenerations,
    /// Load historical metrics for sparklines (on resume)
    LoadMetrics,
    /// Load historical logs from DB (on resume)
    LoadHistoricalLogs,
    /// Load historical samples from DB (on resume)
    LoadHistoricalSamples,
}

/// Current training state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingState {
    /// Waiting for training to start
    #[default]
    Starting,
    /// Training is running
    Running,
    /// Training is paused
    Paused,
    /// Training completed successfully
    Complete,
    /// Training encountered an error
    Error,
    /// Waiting to restart after crash (countdown in progress)
    Restarting,
}

impl TrainingState {
    /// Get display string for state
    pub fn display(&self) -> &'static str {
        match self {
            Self::Starting => "Starting",
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Complete => "Complete",
            Self::Error => "Error",
            Self::Restarting => "Restarting",
        }
    }

    /// Get color for state indicator
    pub fn color(&self) -> ratatui::style::Color {
        use ratatui::style::Color;
        match self {
            Self::Starting => Color::Yellow,
            Self::Running => Color::Green,
            Self::Paused => Color::Yellow,
            Self::Complete => Color::Cyan,
            Self::Error => Color::Red,
            Self::Restarting => Color::Magenta,
        }
    }

    /// Get icon for state
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Starting => "○",
            Self::Running => "▶",
            Self::Paused => "⏸",
            Self::Complete => "✓",
            Self::Error => "✗",
            Self::Restarting => "↻",
        }
    }
}

/// Currently active tab in the right panel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveTab {
    /// Log messages
    #[default]
    Logs,
    /// Generated samples
    Samples,
    /// Training configuration
    Config,
}

impl ActiveTab {
    /// Get tab title
    pub fn title(self) -> &'static str {
        match self {
            Self::Logs => "Logs",
            Self::Samples => "Samples",
            Self::Config => "Config",
        }
    }
}

/// A generated completion sample
#[derive(Debug, Clone)]
pub struct GenerationSample {
    pub index: u32,
    pub prompt: String,
    pub completion: String,
    pub reward: f64,
    pub tokens: u32,
    pub reward_details: Option<std::collections::HashMap<String, f64>>,
}

/// A log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Local>,
}

/// Main application state
pub struct App {
    // Training state
    pub state: TrainingState,
    pub model_name: String,
    pub config: Option<TrainingConfig>,
    /// Training type: "sft" or "grpo" (default: "grpo")
    pub training_type: String,

    // Progress tracking
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_step: u64,
    pub step_in_epoch: u32,
    pub total_steps_in_epoch: u32,

    // Metrics history (for sparklines)
    pub loss_history: VecDeque<f64>,
    // GRPO-specific metric histories
    pub reward_history: VecDeque<f64>,
    pub std_advantage_history: VecDeque<f64>,
    // SFT-specific metric histories
    pub perplexity_history: VecDeque<f64>,
    pub token_accuracy_history: VecDeque<f64>,

    // Current metrics (common)
    pub current_loss: f64,
    pub total_tokens: u64,
    pub generation_time_ms: f64,
    pub training_time_ms: f64,

    // Current metrics (GRPO-specific)
    pub current_reward: f64,
    pub current_std_advantage: f64,
    pub current_std_reward: f64,

    // Current metrics (SFT-specific)
    pub current_perplexity: f64,
    pub current_token_accuracy: f64,

    // Current metrics (Memory)
    pub peak_memory_mb: f64,
    pub active_memory_mb: f64,

    // Aggregated metrics for stats display (GRPO)
    pub best_reward: f64,
    pub reward_sum: f64,
    pub reward_count: u64,

    // Aggregated metrics for stats display (SFT)
    pub best_loss: f64,
    pub loss_sum: f64,
    pub loss_count: u64,

    // Previous values for trend indicators (common)
    pub prev_loss: f64,
    // Previous values for trend indicators (GRPO)
    pub prev_reward: f64,
    pub prev_std_advantage: f64,
    // Previous values for trend indicators (SFT)
    pub prev_perplexity: f64,
    pub prev_token_accuracy: f64,

    // Logs and samples
    pub logs: VecDeque<LogEntry>,
    pub samples: VecDeque<GenerationSample>,
    pub sample_display_mode: SampleDisplayMode,

    // UI state
    pub active_tab: ActiveTab,
    pub log_scroll: u16,
    pub sample_scroll: u16,
    pub config_scroll: u16,
    pub show_help: bool,

    // Timing
    pub start_time: chrono::DateTime<chrono::Local>,
    pub last_checkpoint: Option<String>,
    pub last_checkpoint_step: Option<u64>,

    // Should quit flag
    pub should_quit: bool,

    // Child process has exited
    pub child_exited: bool,

    // Layout info for mouse click detection (updated during render)
    pub tabs_area: Option<(u16, u16, u16, u16)>, // (x, y, width, height)

    // Sample detail popup state
    pub selected_sample: Option<usize>,
    pub sample_detail_scroll: u16,

    // Quit confirmation popup
    pub show_quit_confirm: bool,

    // Settings popup
    pub show_settings: bool,

    // Log level filter (show this level and above)
    pub log_level_filter: LogLevel,

    // Auto-restart state
    /// Number of times the process has been restarted
    pub restart_count: u32,
    /// Countdown seconds until restart (None = not restarting)
    pub restart_countdown: Option<u8>,
    /// Whether auto-restart is enabled
    pub auto_restart_enabled: bool,
    /// Exit code from last process exit
    pub last_exit_code: Option<i32>,

    // Database state
    /// Database reader (if database is open)
    pub db_reader: Option<SyncDbReader>,
    /// Path to the database file
    pub db_path: Option<String>,
    /// Run ID to display (from run or from message)
    pub db_run_id: Option<String>,
    /// Generations loaded from database
    pub db_generations: Vec<GenerationRecord>,
    /// Currently selected generation index in the list
    pub db_selected: usize,
    /// Current filter for database queries
    pub db_filter: GenerationFilter,
    /// Total count of generations matching filter
    pub db_total_count: usize,
    /// Pending database action to process in blocking context
    pub pending_db_action: Option<DbAction>,
    /// Whether ResumeState was received (indicates this is a resumed training run)
    /// Used to trigger loading of historical logs and samples when DatabasePath is received
    pub resume_state_received: bool,

    // Interactive prompt state
    /// Active prompt from training process (blocks UI until answered)
    pub active_prompt: Option<ActivePrompt>,

    // Captured prompt responses for restart
    /// Prompt responses that should be replayed on restart (e.g., training-targets)
    /// Key is prompt ID, value is the response (comma-separated for multi-select)
    pub captured_prompt_responses: std::collections::HashMap<String, String>,
}

/// State for an active interactive prompt
#[derive(Debug, Clone)]
pub struct ActivePrompt {
    /// Unique ID to send back with response
    pub id: String,
    /// Message to display
    pub message: String,
    /// Available choices
    pub choices: Vec<PromptChoice>,
    /// Currently focused index (cursor position)
    pub cursor: usize,
    /// Selected state for each choice (multi-select mode)
    pub selected: Vec<bool>,
    /// Whether this is a multi-select prompt
    pub multi_select: bool,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create a new app instance
    pub fn new() -> Self {
        Self {
            state: TrainingState::Starting,
            model_name: String::new(),
            config: None,
            training_type: "grpo".to_string(), // Default to GRPO
            current_epoch: 0,
            total_epochs: 0,
            current_step: 0,
            step_in_epoch: 0,
            total_steps_in_epoch: 0,
            loss_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            reward_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            std_advantage_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            perplexity_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            token_accuracy_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            // Common metrics
            current_loss: 0.0,
            total_tokens: 0,
            generation_time_ms: 0.0,
            training_time_ms: 0.0,
            // GRPO metrics
            current_reward: 0.0,
            current_std_advantage: 0.0,
            current_std_reward: 0.0,
            // SFT metrics
            current_perplexity: 0.0,
            current_token_accuracy: 0.0,
            // Memory metrics
            peak_memory_mb: 0.0,
            active_memory_mb: 0.0,
            // GRPO aggregates
            best_reward: f64::NEG_INFINITY,
            reward_sum: 0.0,
            reward_count: 0,
            // SFT aggregates
            best_loss: f64::INFINITY,
            loss_sum: 0.0,
            loss_count: 0,
            // Trend indicators
            prev_loss: 0.0,
            prev_reward: 0.0,
            prev_std_advantage: 0.0,
            prev_perplexity: 0.0,
            prev_token_accuracy: 0.0,
            logs: VecDeque::with_capacity(LOG_HISTORY),
            samples: VecDeque::with_capacity(SAMPLE_HISTORY),
            sample_display_mode: SampleDisplayMode::default(),
            active_tab: ActiveTab::default(),
            log_scroll: 0,
            sample_scroll: 0,
            config_scroll: 0,
            show_help: false,
            start_time: chrono::Local::now(),
            last_checkpoint: None,
            last_checkpoint_step: None,
            should_quit: false,
            child_exited: false,
            tabs_area: None,
            selected_sample: None,
            sample_detail_scroll: 0,
            show_quit_confirm: false,
            show_settings: false,
            log_level_filter: LogLevel::Debug,
            restart_count: 0,
            restart_countdown: None,
            auto_restart_enabled: true,
            last_exit_code: None,
            // Database state
            db_reader: None,
            db_path: None,
            db_run_id: None,
            db_generations: Vec::new(),
            db_selected: 0,
            db_filter: GenerationFilter::default(),
            db_total_count: 0,
            pending_db_action: None,
            resume_state_received: false,
            // Interactive prompt
            active_prompt: None,
            // Captured prompt responses for restart
            captured_prompt_responses: std::collections::HashMap::new(),
        }
    }

    /// Handle an incoming training message
    pub fn handle_message(&mut self, msg: TrainingMessage) {
        debug!(msg_type = ?std::mem::discriminant(&msg), "handle_message");
        match msg {
            TrainingMessage::Init { model, config } => {
                debug!(%model, "Init message received");
                self.model_name = model;
                self.total_epochs = config.num_epochs.unwrap_or(1);
                // Extract training type, default to "grpo" if not specified
                self.training_type = config
                    .training_type
                    .clone()
                    .unwrap_or_else(|| "grpo".to_string());
                self.config = Some(config);
                self.state = TrainingState::Running;

                // Ensure active tab is valid for the training type
                // SFT mode doesn't have Samples or Database tabs
                let available_tabs = self.get_available_tabs();
                if !available_tabs.contains(&self.active_tab) {
                    self.active_tab = available_tabs[0];
                }

                let training_label = if self.training_type == "sft" {
                    "SFT"
                } else {
                    "GRPO"
                };
                self.add_log(
                    LogLevel::Info,
                    format!("{} training initialized", training_label),
                );
            }

            TrainingMessage::EpochStart {
                epoch,
                total_epochs,
                num_batches,
            } => {
                // Preserve step_in_epoch if this is the same epoch we restored from resume_state
                // This prevents resetting progress when resuming mid-epoch
                let should_reset_step = !self.resume_state_received || epoch != self.current_epoch;

                self.current_epoch = epoch;
                self.total_epochs = total_epochs;
                self.total_steps_in_epoch = num_batches;

                if should_reset_step {
                    self.step_in_epoch = 0;
                }

                self.add_log(
                    LogLevel::Info,
                    format!("Epoch {epoch}/{total_epochs} started ({num_batches} batches)"),
                );
            }

            TrainingMessage::Step {
                step,
                loss,
                mean_reward,
                std_reward,
                std_advantage,
                perplexity,
                token_accuracy,
                total_tokens,
                generation_time_ms,
                training_time_ms,
                peak_memory_mb,
                active_memory_mb,
            } => {
                // Store previous values for trend indicators
                self.prev_loss = self.current_loss;
                self.prev_reward = self.current_reward;
                self.prev_std_advantage = self.current_std_advantage;
                self.prev_perplexity = self.current_perplexity;
                self.prev_token_accuracy = self.current_token_accuracy;

                // Update common metrics
                self.current_step = step;
                self.step_in_epoch += 1;
                self.current_loss = loss;
                self.total_tokens += total_tokens as u64;
                self.generation_time_ms = generation_time_ms.unwrap_or(0.0);
                self.training_time_ms = training_time_ms.unwrap_or(0.0);

                // Update memory metrics
                self.peak_memory_mb = peak_memory_mb.unwrap_or(0.0);
                self.active_memory_mb = active_memory_mb.unwrap_or(0.0);

                // Update loss history (common to both SFT and GRPO)
                if self.loss_history.len() >= SPARKLINE_HISTORY {
                    self.loss_history.pop_front();
                }
                self.loss_history.push_back(loss);

                // Track training type-specific metrics
                if self.training_type == "sft" {
                    // SFT metrics
                    let ppl = perplexity.unwrap_or_else(|| loss.exp());
                    self.current_perplexity = ppl;
                    self.current_token_accuracy = token_accuracy.unwrap_or(0.0);

                    // Update SFT sparkline histories
                    if self.perplexity_history.len() >= SPARKLINE_HISTORY {
                        self.perplexity_history.pop_front();
                    }
                    self.perplexity_history.push_back(ppl);

                    if let Some(acc) = token_accuracy {
                        if self.token_accuracy_history.len() >= SPARKLINE_HISTORY {
                            self.token_accuracy_history.pop_front();
                        }
                        self.token_accuracy_history.push_back(acc);
                    }

                    // Track best and average loss (SFT)
                    if loss < self.best_loss {
                        self.best_loss = loss;
                    }
                    self.loss_sum += loss;
                    self.loss_count += 1;
                } else {
                    // GRPO metrics
                    let reward = mean_reward.unwrap_or(0.0);
                    let std_adv = std_advantage.unwrap_or(0.0);

                    self.current_reward = reward;
                    self.current_std_reward = std_reward.unwrap_or(0.0);
                    self.current_std_advantage = std_adv;

                    // Update GRPO sparkline histories
                    if self.reward_history.len() >= SPARKLINE_HISTORY {
                        self.reward_history.pop_front();
                    }
                    self.reward_history.push_back(reward);

                    if self.std_advantage_history.len() >= SPARKLINE_HISTORY {
                        self.std_advantage_history.pop_front();
                    }
                    self.std_advantage_history.push_back(std_adv);

                    // Track best and average reward (GRPO)
                    if reward > self.best_reward {
                        self.best_reward = reward;
                    }
                    self.reward_sum += reward;
                    self.reward_count += 1;
                }
            }

            TrainingMessage::Generation {
                index,
                prompt,
                completion,
                reward,
                tokens,
                reward_details,
            } => {
                if self.samples.len() >= SAMPLE_HISTORY {
                    self.samples.pop_front();
                }
                self.samples.push_back(GenerationSample {
                    index,
                    prompt,
                    completion,
                    reward,
                    tokens,
                    reward_details,
                });
            }

            TrainingMessage::Checkpoint { path, step } => {
                self.last_checkpoint = Some(path.clone());
                self.last_checkpoint_step = Some(step);
                self.add_log(LogLevel::Info, format!("Checkpoint saved: {path}"));
            }

            TrainingMessage::EpochEnd {
                epoch,
                avg_loss,
                avg_reward,
                epoch_time_secs,
            } => {
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Epoch {epoch} complete: loss={avg_loss:.4}, reward={avg_reward:.4}, time={epoch_time_secs:.1}s"
                    ),
                );
            }

            TrainingMessage::Complete {
                total_steps,
                total_time_secs,
            } => {
                self.state = TrainingState::Complete;
                let mins = total_time_secs / 60.0;
                self.add_log(
                    LogLevel::Info,
                    format!("Training complete: {total_steps} steps in {mins:.1} minutes"),
                );
            }

            TrainingMessage::Log { level, message } => {
                self.add_log(level, message);
            }

            TrainingMessage::Paused { step } => {
                self.state = TrainingState::Paused;
                self.add_log(LogLevel::Info, format!("Training paused at step {step}"));
            }

            TrainingMessage::Resumed { step } => {
                self.state = TrainingState::Running;
                self.add_log(LogLevel::Info, format!("Training resumed at step {step}"));
            }

            TrainingMessage::Status { phase, message } => {
                // Update model name during loading phase for better feedback
                if phase == "loading" {
                    self.model_name = message.clone();
                }
                self.add_log(LogLevel::Info, message);
            }

            TrainingMessage::DatabasePath {
                path,
                run_id,
                run_name,
            } => {
                // Open database and set run ID
                if let Err(e) = self.open_database(&path) {
                    self.add_log(LogLevel::Error, e);
                } else {
                    self.set_database_run_id(run_id.clone());
                    if let Some(name) = &run_name {
                        self.add_log(LogLevel::Info, format!("Database run: {name} ({run_id})"));
                    } else {
                        self.add_log(LogLevel::Info, format!("Database run: {run_id}"));
                    }
                    // Handle resume state restoration
                    if self.resume_state_received {
                        // ResumeState already populated sparklines, so load logs and samples
                        if self.logs.is_empty() && self.pending_db_action.is_none() {
                            self.pending_db_action = Some(DbAction::LoadHistoricalLogs);
                        }
                    } else if self.loss_history.is_empty() && self.pending_db_action.is_none() {
                        // Fallback: load metrics if sparklines are empty
                        self.pending_db_action = Some(DbAction::LoadMetrics);
                    }
                }
            }

            TrainingMessage::Prompt {
                id,
                message,
                choices,
                default,
                multi_select,
            } => {
                debug!(%id, %message, num_choices = choices.len(), multi_select, "Prompt received");
                let num_choices = choices.len();
                let mut selected = vec![false; num_choices];
                // Initialize cursor to first default index (for single-select)
                let mut cursor = 0usize;

                // Apply default selections
                if let Some(ref defaults) = default {
                    for &idx in defaults {
                        if idx < num_choices {
                            selected[idx] = true;
                            // For single-select, set cursor to first default
                            if cursor == 0 {
                                cursor = idx;
                            }
                        }
                    }
                }

                self.active_prompt = Some(ActivePrompt {
                    id,
                    message,
                    choices,
                    cursor,
                    selected,
                    multi_select,
                });
            }

            TrainingMessage::ResumeState {
                step,
                epoch,
                total_epochs,
                step_in_epoch,
                total_steps_in_epoch,
                metrics_history,
                aggregates,
            } => {
                info!(
                    "Restoring TUI state: step={}, epoch={}/{}, batch={}/{}, {} historical metrics",
                    step,
                    epoch,
                    total_epochs,
                    step_in_epoch,
                    total_steps_in_epoch,
                    metrics_history.len()
                );
                self.restore_from_resume(
                    step,
                    epoch,
                    total_epochs,
                    step_in_epoch,
                    total_steps_in_epoch,
                    metrics_history,
                    aggregates,
                );
            }
        }
    }

    /// Restore TUI state from resume data
    ///
    /// Called when training script sends ResumeState message to restore
    /// sparklines, aggregates, and progress state.
    #[allow(clippy::too_many_arguments)]
    fn restore_from_resume(
        &mut self,
        step: u64,
        epoch: u32,
        total_epochs: u32,
        step_in_epoch: u32,
        total_steps_in_epoch: u32,
        metrics_history: Vec<ResumeMetric>,
        aggregates: ResumeAggregates,
    ) {
        // Restore progress state
        self.current_step = step;
        self.current_epoch = epoch;
        self.total_epochs = total_epochs;

        // Restore batch progress within epoch
        if total_steps_in_epoch > 0 {
            self.step_in_epoch = step_in_epoch;
            self.total_steps_in_epoch = total_steps_in_epoch;
        }

        // Clear existing histories and restore from metrics_history (oldest first)
        self.loss_history.clear();
        self.reward_history.clear();
        self.std_advantage_history.clear();
        self.perplexity_history.clear();
        self.token_accuracy_history.clear();

        // Track last timing values from metrics
        let mut last_generation_time_ms: Option<f64> = None;
        let mut last_training_time_ms: Option<f64> = None;

        for metric in metrics_history {
            // Only keep up to SPARKLINE_HISTORY entries
            if self.loss_history.len() >= SPARKLINE_HISTORY {
                self.loss_history.pop_front();
            }
            self.loss_history.push_back(metric.loss);

            if self.reward_history.len() >= SPARKLINE_HISTORY {
                self.reward_history.pop_front();
            }
            self.reward_history.push_back(metric.mean_reward);

            if self.std_advantage_history.len() >= SPARKLINE_HISTORY {
                self.std_advantage_history.pop_front();
            }
            self.std_advantage_history.push_back(metric.std_advantage);

            // SFT-specific metrics
            if let Some(ppl) = metric.perplexity {
                if self.perplexity_history.len() >= SPARKLINE_HISTORY {
                    self.perplexity_history.pop_front();
                }
                self.perplexity_history.push_back(ppl);
            }
            if let Some(acc) = metric.token_accuracy {
                if self.token_accuracy_history.len() >= SPARKLINE_HISTORY {
                    self.token_accuracy_history.pop_front();
                }
                self.token_accuracy_history.push_back(acc);
            }

            // Track timing from last metric
            if metric.generation_time_ms.is_some() {
                last_generation_time_ms = metric.generation_time_ms;
            }
            if metric.training_time_ms.is_some() {
                last_training_time_ms = metric.training_time_ms;
            }
        }

        // Update current values from last entry
        if let Some(&loss) = self.loss_history.back() {
            self.current_loss = loss;
            self.prev_loss = loss;
        }
        if let Some(&reward) = self.reward_history.back() {
            self.current_reward = reward;
            self.prev_reward = reward;
        }
        if let Some(&adv) = self.std_advantage_history.back() {
            self.current_std_advantage = adv;
            self.prev_std_advantage = adv;
        }
        if let Some(&ppl) = self.perplexity_history.back() {
            self.current_perplexity = ppl;
            self.prev_perplexity = ppl;
        }
        if let Some(&acc) = self.token_accuracy_history.back() {
            self.current_token_accuracy = acc;
            self.prev_token_accuracy = acc;
        }

        // Restore aggregates
        self.best_reward = aggregates.best_reward;
        self.reward_sum = aggregates.avg_reward * aggregates.reward_count as f64;
        self.reward_count = aggregates.reward_count as u64;
        self.best_loss = aggregates.best_loss;
        self.loss_sum = aggregates.avg_loss * aggregates.loss_count as f64;
        self.loss_count = aggregates.loss_count as u64;
        self.total_tokens = aggregates.total_tokens as u64;

        // Restore timing from last metric entry, or fall back to aggregates average
        self.generation_time_ms =
            last_generation_time_ms.unwrap_or(aggregates.avg_generation_time_ms);
        self.training_time_ms = last_training_time_ms.unwrap_or(aggregates.avg_training_time_ms);

        // Mark that we received resume state - this triggers historical log/sample loading
        // when DatabasePath is processed
        self.resume_state_received = true;

        self.add_log(
            LogLevel::Info,
            format!(
                "Restored {} historical metrics from database (step {}, epoch {}/{})",
                self.loss_history.len(),
                step,
                epoch,
                total_epochs
            ),
        );
    }

    /// Restore historical logs from database
    ///
    /// Called when resuming to populate the logs panel with previous log entries.
    /// Logs are prepended (oldest first) to maintain chronological order.
    pub fn restore_historical_logs(&mut self, logs: Vec<mlx_db::LogRecord>) {
        let error_count_before = self
            .logs
            .iter()
            .filter(|e| e.level == LogLevel::Error)
            .count();
        debug!(
            "restore_historical_logs called. Current logs: {}, errors: {}, incoming: {}",
            self.logs.len(),
            error_count_before,
            logs.len()
        );

        // Insert logs at the front (they're ordered oldest first from DB)
        for log in logs.into_iter().rev() {
            let level = match log.level.as_str() {
                "debug" => LogLevel::Debug,
                "info" => LogLevel::Info,
                "warn" => LogLevel::Warn,
                "error" => LogLevel::Error,
                _ => LogLevel::Info,
            };
            // Insert at front to maintain order (newest at back)
            self.logs.push_front(LogEntry {
                level,
                message: log.message,
                timestamp: chrono::DateTime::from_timestamp_millis(log.created_at)
                    .map(|dt| dt.with_timezone(&chrono::Local))
                    .unwrap_or_else(chrono::Local::now),
            });
        }
        // Trim to max size - but pop from front (oldest) not back (newest)!
        // This preserves the most recent logs including any crash errors
        while self.logs.len() > LOG_HISTORY {
            self.logs.pop_front();
        }

        let error_count_after = self
            .logs
            .iter()
            .filter(|e| e.level == LogLevel::Error)
            .count();
        debug!(
            "restore_historical_logs done. Final logs: {}, errors: {}",
            self.logs.len(),
            error_count_after
        );
    }

    /// Restore historical samples from database generations
    ///
    /// Called when resuming to populate the samples panel with previous generations.
    pub fn restore_historical_samples(&mut self, generations: Vec<GenerationRecord>) {
        // Insert samples at the front (they're ordered most recent first from DB)
        for record in generations.into_iter().rev() {
            if self.samples.len() >= SAMPLE_HISTORY {
                self.samples.pop_back();
            }
            self.samples.push_front(GenerationSample {
                index: record.id.unwrap_or(0) as u32,
                prompt: record.prompt,
                completion: record.completion_text,
                reward: record.reward,
                tokens: record.num_tokens as u32,
                reward_details: None,
            });
        }
    }

    /// Add a log entry
    pub fn add_log(&mut self, level: LogLevel, message: String) {
        // Emit to tracing (writes to file for crash recovery)
        match level {
            LogLevel::Debug => debug!(target: "ui", "{}", message),
            LogLevel::Info => info!(target: "ui", "{}", message),
            LogLevel::Warn => warn!(target: "ui", "{}", message),
            LogLevel::Error => error!(target: "ui", "{}", message),
        }

        // Debug: trace error logs specifically
        if level == LogLevel::Error {
            debug!(
                "Adding ERROR log to UI: '{}' (total logs before: {})",
                message,
                self.logs.len()
            );
        }

        // NOTE: Database log persistence removed - block_on cannot be called
        // from within the async tokio runtime (causes nested runtime panic).
        // Logs are persisted via tracing to file instead.

        // Keep in memory for UI display
        if self.logs.len() >= LOG_HISTORY {
            self.logs.pop_front();
        }
        self.logs.push_back(LogEntry {
            level,
            message,
            timestamp: chrono::Local::now(),
        });

        // Debug: trace error count after adding
        if level == LogLevel::Error {
            let error_count = self
                .logs
                .iter()
                .filter(|e| e.level == LogLevel::Error)
                .count();
            debug!(
                "After adding: total logs = {}, error logs = {}",
                self.logs.len(),
                error_count
            );
        }

        // Auto-scroll to bottom
        self.log_scroll = self.logs.len().saturating_sub(1) as u16;
    }

    /// Get the list of available tabs based on training type
    /// SFT mode only shows Logs and Config (no Samples)
    pub fn get_available_tabs(&self) -> Vec<ActiveTab> {
        if self.training_type == "sft" {
            vec![ActiveTab::Logs, ActiveTab::Config]
        } else {
            vec![ActiveTab::Logs, ActiveTab::Samples, ActiveTab::Config]
        }
    }

    /// Toggle to next tab (respects training type)
    pub fn next_tab(&mut self) {
        let available = self.get_available_tabs();
        if let Some(idx) = available.iter().position(|t| *t == self.active_tab) {
            let next_idx = (idx + 1) % available.len();
            self.switch_to_tab(available[next_idx]);
        } else {
            // Current tab not in available list, switch to first available
            self.switch_to_tab(available[0]);
        }
    }

    /// Toggle to previous tab (respects training type)
    pub fn prev_tab(&mut self) {
        let available = self.get_available_tabs();
        if let Some(idx) = available.iter().position(|t| *t == self.active_tab) {
            let prev_idx = if idx == 0 {
                available.len() - 1
            } else {
                idx - 1
            };
            self.switch_to_tab(available[prev_idx]);
        } else {
            // Current tab not in available list, switch to last available
            self.switch_to_tab(available[available.len() - 1]);
        }
    }

    /// Switch to a specific tab
    pub fn switch_to_tab(&mut self, tab: ActiveTab) {
        self.active_tab = tab;
    }

    /// Scroll up in current tab
    pub fn scroll_up(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => self.log_scroll = self.log_scroll.saturating_sub(1),
            ActiveTab::Samples => self.sample_scroll = self.sample_scroll.saturating_sub(1),
            ActiveTab::Config => self.config_scroll = self.config_scroll.saturating_sub(1),
        }
    }

    /// Scroll down in current tab
    pub fn scroll_down(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => {
                let max = self.logs.len().saturating_sub(1) as u16;
                self.log_scroll = (self.log_scroll + 1).min(max);
            }
            ActiveTab::Samples => {
                let max = self.samples.len().saturating_sub(1) as u16;
                self.sample_scroll = (self.sample_scroll + 1).min(max);
            }
            ActiveTab::Config => {
                self.config_scroll += 1;
            }
        }
    }

    /// Page up in current tab
    pub fn page_up(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => self.log_scroll = self.log_scroll.saturating_sub(10),
            ActiveTab::Samples => self.sample_scroll = self.sample_scroll.saturating_sub(10),
            ActiveTab::Config => self.config_scroll = self.config_scroll.saturating_sub(10),
        }
    }

    /// Page down in current tab
    pub fn page_down(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => {
                let max = self.logs.len().saturating_sub(1) as u16;
                self.log_scroll = (self.log_scroll + 10).min(max);
            }
            ActiveTab::Samples => {
                let max = self.samples.len().saturating_sub(1) as u16;
                self.sample_scroll = (self.sample_scroll + 10).min(max);
            }
            ActiveTab::Config => {
                self.config_scroll += 10;
            }
        }
    }

    /// Scroll to top of current tab
    pub fn scroll_to_top(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => self.log_scroll = 0,
            ActiveTab::Samples => self.sample_scroll = 0,
            ActiveTab::Config => self.config_scroll = 0,
        }
    }

    /// Scroll to bottom of current tab
    pub fn scroll_to_bottom(&mut self) {
        match self.active_tab {
            ActiveTab::Logs => {
                self.log_scroll = self.logs.len().saturating_sub(1) as u16;
            }
            ActiveTab::Samples => {
                self.sample_scroll = self.samples.len().saturating_sub(1) as u16;
            }
            ActiveTab::Config => {
                // Config doesn't have a known length, just scroll far
                self.config_scroll = u16::MAX / 2;
            }
        }
    }

    /// Toggle help overlay
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Cycle sample display mode
    pub fn cycle_sample_mode(&mut self) {
        self.sample_display_mode = self.sample_display_mode.next();
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> chrono::Duration {
        chrono::Local::now() - self.start_time
    }

    /// Format elapsed time as HH:MM:SS
    pub fn elapsed_str(&self) -> String {
        let elapsed = self.elapsed();
        let hours = elapsed.num_hours();
        let mins = elapsed.num_minutes() % 60;
        let secs = elapsed.num_seconds() % 60;
        format!("{hours}:{mins:02}:{secs:02}")
    }

    /// Get epoch progress as fraction (0.0 to 1.0)
    pub fn epoch_progress(&self) -> f64 {
        if self.total_epochs == 0 {
            return 0.0;
        }
        self.current_epoch as f64 / self.total_epochs as f64
    }

    /// Get step progress within epoch as fraction (0.0 to 1.0)
    pub fn step_progress(&self) -> f64 {
        if self.total_steps_in_epoch == 0 {
            return 0.0;
        }
        self.step_in_epoch as f64 / self.total_steps_in_epoch as f64
    }

    /// Get total ms per step
    pub fn ms_per_step(&self) -> f64 {
        self.generation_time_ms + self.training_time_ms
    }

    /// Get average reward (GRPO)
    pub fn avg_reward(&self) -> f64 {
        if self.reward_count == 0 {
            0.0
        } else {
            self.reward_sum / self.reward_count as f64
        }
    }

    /// Get average loss (SFT)
    pub fn avg_loss(&self) -> f64 {
        if self.loss_count == 0 {
            0.0
        } else {
            self.loss_sum / self.loss_count as f64
        }
    }

    /// Get tokens per second
    pub fn tokens_per_sec(&self) -> f64 {
        let elapsed_secs = self.elapsed().num_seconds() as f64;
        if elapsed_secs <= 0.0 {
            0.0
        } else {
            self.total_tokens as f64 / elapsed_secs
        }
    }

    /// Get estimated time remaining as formatted string
    pub fn eta_str(&self) -> String {
        // Calculate based on steps remaining
        let total_steps = self.total_epochs as u64 * self.total_steps_in_epoch as u64;
        let completed_steps = (self.current_epoch.saturating_sub(1)) as u64
            * self.total_steps_in_epoch as u64
            + self.step_in_epoch as u64;

        if completed_steps == 0 || total_steps == 0 {
            return "calculating...".to_string();
        }

        let remaining_steps = total_steps.saturating_sub(completed_steps);
        let ms_per_step = self.ms_per_step();

        if ms_per_step <= 0.0 {
            return "calculating...".to_string();
        }

        let remaining_ms = remaining_steps as f64 * ms_per_step;
        let remaining_secs = (remaining_ms / 1000.0) as i64;

        let hours = remaining_secs / 3600;
        let mins = (remaining_secs % 3600) / 60;
        let secs = remaining_secs % 60;

        format!("{hours}:{mins:02}:{secs:02}")
    }

    /// Get trend indicator for a metric
    pub fn trend_indicator(current: f64, previous: f64) -> &'static str {
        let threshold = 0.0001; // Small threshold to avoid noise
        if current > previous + threshold {
            "↑"
        } else if current < previous - threshold {
            "↓"
        } else {
            "→"
        }
    }

    /// Prepare for restart - reset process state while keeping metrics
    /// Returns the old db_reader if any, which must be dropped via spawn_blocking
    pub fn prepare_for_restart(&mut self) -> Option<SyncDbReader> {
        let error_count_before = self
            .logs
            .iter()
            .filter(|e| e.level == LogLevel::Error)
            .count();
        debug!(
            "prepare_for_restart called. Logs before: {}, errors: {}",
            self.logs.len(),
            error_count_before
        );

        self.state = TrainingState::Starting;
        self.child_exited = false;
        self.restart_countdown = None;
        self.restart_count += 1;

        // Take database connection for safe drop via spawn_blocking
        // The new training process will send a DatabasePath message
        let old_reader = self.db_reader.take();
        if old_reader.is_some() {
            debug!("Closing database connection for restart");
        }

        // Keep metrics history, logs, and samples for continuity
        self.add_log(
            LogLevel::Info,
            format!("Restarting training (attempt #{})...", self.restart_count),
        );

        let error_count_after = self
            .logs
            .iter()
            .filter(|e| e.level == LogLevel::Error)
            .count();
        debug!(
            "prepare_for_restart done. Logs after: {}, errors: {}",
            self.logs.len(),
            error_count_after
        );

        old_reader
    }

    /// Start restart countdown
    pub fn start_restart_countdown(&mut self, seconds: u8) {
        self.state = TrainingState::Restarting;
        self.restart_countdown = Some(seconds);
    }

    /// Decrement restart countdown, returns true if countdown reached zero
    pub fn tick_restart_countdown(&mut self) -> bool {
        if let Some(count) = self.restart_countdown {
            if count <= 1 {
                self.restart_countdown = Some(0);
                true
            } else {
                self.restart_countdown = Some(count - 1);
                false
            }
        } else {
            false
        }
    }

    /// Cancel restart countdown
    pub fn cancel_restart(&mut self) {
        self.restart_countdown = None;
        self.state = TrainingState::Error;
        self.auto_restart_enabled = false;
        self.add_log(LogLevel::Info, "Auto-restart cancelled".to_string());
    }

    // === Database Methods ===

    /// Open a database file
    /// WARNING: This calls block_on internally, do NOT call from async context!
    /// Use spawn_blocking in main.rs instead.
    pub fn open_database(&mut self, path: &str) -> Result<(), String> {
        debug!(%path, "open_database");
        match SyncDbReader::open(path) {
            Ok(reader) => {
                debug!("SyncDbReader::open succeeded");
                self.db_reader = Some(reader);
                self.db_path = Some(path.to_string());
                self.add_log(LogLevel::Info, format!("Opened database: {path}"));
                // Request data fetch (will be processed in event loop via spawn_blocking)
                self.pending_db_action = Some(DbAction::RefreshGenerations);
                debug!("open_database complete, refresh pending");
                Ok(())
            }
            Err(e) => {
                debug!(%e, "SyncDbReader::open failed");
                let msg = format!("Failed to open database: {e}");
                self.add_log(LogLevel::Error, msg.clone());
                Err(msg)
            }
        }
    }

    /// Set the database run ID (from training process message)
    pub fn set_database_run_id(&mut self, run_id: String) {
        debug!(%run_id, "set_database_run_id");
        self.db_run_id = Some(run_id);
        // Request data fetch (will be processed in event loop via spawn_blocking)
        self.pending_db_action = Some(DbAction::RefreshGenerations);
    }

    /// Store generations result (called after spawn_blocking fetch)
    pub fn set_generations(&mut self, generations: Vec<GenerationRecord>, total: usize) {
        self.db_total_count = total;
        self.db_generations = generations;
        if self.db_selected >= self.db_generations.len() && !self.db_generations.is_empty() {
            self.db_selected = self.db_generations.len() - 1;
        }
    }

    /// Restore metrics history from database query results
    /// Called on resume to populate sparklines with historical data
    pub fn restore_metrics_history(&mut self, metrics: Vec<(f64, f64, f64)>) {
        // metrics is Vec<(loss, mean_reward, std_advantage)> ordered by step
        self.loss_history.clear();
        self.reward_history.clear();
        self.std_advantage_history.clear();

        // Take last SPARKLINE_HISTORY entries
        let start = metrics.len().saturating_sub(SPARKLINE_HISTORY);
        for (loss, reward, std_adv) in metrics.into_iter().skip(start) {
            self.loss_history.push_back(loss);
            self.reward_history.push_back(reward);
            self.std_advantage_history.push_back(std_adv);
        }

        // Update current values to last entry
        if let Some(&loss) = self.loss_history.back() {
            self.current_loss = loss;
        }
        if let Some(&reward) = self.reward_history.back() {
            self.current_reward = reward;
        }
        if let Some(&adv) = self.std_advantage_history.back() {
            self.current_std_advantage = adv;
        }

        self.add_log(
            crate::messages::LogLevel::Info,
            format!(
                "Restored {} historical metrics from database",
                self.loss_history.len()
            ),
        );
    }

    /// Take pending db action (clears it)
    pub fn take_pending_db_action(&mut self) -> Option<DbAction> {
        self.pending_db_action.take()
    }

    // === Prompt Methods ===

    /// Move cursor up in active prompt
    pub fn prompt_select_prev(&mut self) {
        if let Some(ref mut prompt) = self.active_prompt {
            prompt.cursor = prompt.cursor.saturating_sub(1);
        }
    }

    /// Move cursor down in active prompt
    pub fn prompt_select_next(&mut self) {
        if let Some(ref mut prompt) = self.active_prompt {
            let max = prompt.choices.len().saturating_sub(1);
            prompt.cursor = (prompt.cursor + 1).min(max);
        }
    }

    /// Toggle selection at cursor (for multi-select mode)
    pub fn prompt_toggle(&mut self) {
        if let Some(ref mut prompt) = self.active_prompt
            && prompt.multi_select
            && let Some(selected) = prompt.selected.get_mut(prompt.cursor)
        {
            *selected = !*selected;
        }
    }

    /// Get the selected value(s) and clear the prompt
    /// Returns (id, value) where value is comma-separated for multi-select
    /// Also captures certain prompts (like training-targets) for replay on restart
    pub fn prompt_confirm(&mut self) -> Option<(String, String)> {
        let prompt = self.active_prompt.take()?;

        let (id, value) = if prompt.multi_select {
            // Collect all selected values
            let values: Vec<&str> = prompt
                .choices
                .iter()
                .zip(prompt.selected.iter())
                .filter(|(_, selected)| **selected)
                .map(|(choice, _)| choice.value.as_str())
                .collect();
            (prompt.id, values.join(","))
        } else {
            // Single-select: return the cursor position's value
            let value = prompt.choices.get(prompt.cursor)?.value.clone();
            (prompt.id, value)
        };

        // Capture all prompt responses for replay on restart
        // This allows the TUI to remember user selections (like training targets)
        // and automatically replay them when the training script restarts
        if !value.is_empty() {
            self.captured_prompt_responses
                .insert(id.clone(), value.clone());
        }

        Some((id, value))
    }

    /// Check if there's an active prompt blocking UI
    pub fn has_active_prompt(&self) -> bool {
        self.active_prompt.is_some()
    }
}
