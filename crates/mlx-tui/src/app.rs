//! Application state and logic
//!
//! Contains the main App struct that holds all TUI state and handles
//! incoming messages from the training process.

use std::collections::VecDeque;

use crate::commands::SampleDisplayMode;
use crate::messages::{LogLevel, TrainingConfig, TrainingMessage};

/// Maximum number of data points to keep for sparklines
const SPARKLINE_HISTORY: usize = 60;
/// Maximum number of log entries to keep
const LOG_HISTORY: usize = 500;
/// Maximum number of generation samples to keep
const SAMPLE_HISTORY: usize = 50;

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
    /// Cycle to next tab
    pub fn next(self) -> Self {
        match self {
            Self::Logs => Self::Samples,
            Self::Samples => Self::Config,
            Self::Config => Self::Logs,
        }
    }

    /// Cycle to previous tab
    pub fn prev(self) -> Self {
        match self {
            Self::Logs => Self::Config,
            Self::Samples => Self::Logs,
            Self::Config => Self::Samples,
        }
    }

    /// Get tab index (0-based)
    pub fn index(self) -> usize {
        match self {
            Self::Logs => 0,
            Self::Samples => 1,
            Self::Config => 2,
        }
    }

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

    // Progress tracking
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_step: u64,
    pub step_in_epoch: u32,
    pub total_steps_in_epoch: u32,

    // Metrics history (for sparklines)
    pub loss_history: VecDeque<f64>,
    pub reward_history: VecDeque<f64>,
    pub advantage_history: VecDeque<f64>,

    // Current metrics
    pub current_loss: f64,
    pub current_reward: f64,
    pub current_advantage: f64,
    pub current_std_reward: f64,
    pub total_tokens: u64,
    pub generation_time_ms: f64,
    pub training_time_ms: f64,

    // Aggregated metrics for stats display
    pub best_reward: f64,
    pub reward_sum: f64,
    pub reward_count: u64,

    // Previous values for trend indicators
    pub prev_loss: f64,
    pub prev_reward: f64,
    pub prev_advantage: f64,

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
            current_epoch: 0,
            total_epochs: 0,
            current_step: 0,
            step_in_epoch: 0,
            total_steps_in_epoch: 0,
            loss_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            reward_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            advantage_history: VecDeque::with_capacity(SPARKLINE_HISTORY),
            current_loss: 0.0,
            current_reward: 0.0,
            current_advantage: 0.0,
            current_std_reward: 0.0,
            total_tokens: 0,
            generation_time_ms: 0.0,
            training_time_ms: 0.0,
            best_reward: f64::NEG_INFINITY,
            reward_sum: 0.0,
            reward_count: 0,
            prev_loss: 0.0,
            prev_reward: 0.0,
            prev_advantage: 0.0,
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
            log_level_filter: LogLevel::Info,
        }
    }

    /// Handle an incoming training message
    pub fn handle_message(&mut self, msg: TrainingMessage) {
        match msg {
            TrainingMessage::Init { model, config } => {
                self.model_name = model;
                self.total_epochs = config.num_epochs.unwrap_or(1);
                self.config = Some(config);
                self.state = TrainingState::Running;
                self.add_log(LogLevel::Info, "Training initialized".to_string());
            }

            TrainingMessage::EpochStart {
                epoch,
                total_epochs,
                num_batches,
            } => {
                self.current_epoch = epoch;
                self.total_epochs = total_epochs;
                self.total_steps_in_epoch = num_batches;
                self.step_in_epoch = 0;
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
                mean_advantage,
                total_tokens,
                generation_time_ms,
                training_time_ms,
            } => {
                // Store previous values for trend indicators
                self.prev_loss = self.current_loss;
                self.prev_reward = self.current_reward;
                self.prev_advantage = self.current_advantage;

                self.current_step = step;
                self.step_in_epoch += 1;
                self.current_loss = loss;
                self.current_reward = mean_reward;
                self.current_std_reward = std_reward;
                self.current_advantage = mean_advantage;
                self.total_tokens += total_tokens as u64;
                self.generation_time_ms = generation_time_ms;
                self.training_time_ms = training_time_ms;

                // Track best and average reward
                if mean_reward > self.best_reward {
                    self.best_reward = mean_reward;
                }
                self.reward_sum += mean_reward;
                self.reward_count += 1;

                // Update sparkline history
                self.push_history(&mut self.loss_history.clone(), loss);
                self.push_history(&mut self.reward_history.clone(), mean_reward);
                self.push_history(&mut self.advantage_history.clone(), mean_advantage);
                // Actually mutate (workaround for borrow checker)
                if self.loss_history.len() >= SPARKLINE_HISTORY {
                    self.loss_history.pop_front();
                }
                self.loss_history.push_back(loss);
                if self.reward_history.len() >= SPARKLINE_HISTORY {
                    self.reward_history.pop_front();
                }
                self.reward_history.push_back(mean_reward);
                if self.advantage_history.len() >= SPARKLINE_HISTORY {
                    self.advantage_history.pop_front();
                }
                self.advantage_history.push_back(mean_advantage);
            }

            TrainingMessage::Generation {
                index,
                prompt,
                completion,
                reward,
                tokens,
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
        }
    }

    /// Add a log entry
    fn add_log(&mut self, level: LogLevel, message: String) {
        if self.logs.len() >= LOG_HISTORY {
            self.logs.pop_front();
        }
        self.logs.push_back(LogEntry {
            level,
            message,
            timestamp: chrono::Local::now(),
        });
        // Auto-scroll to bottom
        self.log_scroll = self.logs.len().saturating_sub(1) as u16;
    }

    /// Push a value to a history deque, maintaining max size
    fn push_history(&self, history: &mut VecDeque<f64>, value: f64) {
        if history.len() >= SPARKLINE_HISTORY {
            history.pop_front();
        }
        history.push_back(value);
    }

    /// Toggle to next tab
    pub fn next_tab(&mut self) {
        self.active_tab = self.active_tab.next();
    }

    /// Toggle to previous tab
    pub fn prev_tab(&mut self) {
        self.active_tab = self.active_tab.prev();
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

    /// Get average reward
    pub fn avg_reward(&self) -> f64 {
        if self.reward_count == 0 {
            0.0
        } else {
            self.reward_sum / self.reward_count as f64
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
}
