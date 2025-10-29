//! Control commands sent from TUI to training process
//!
//! These commands are sent via stdin to control the training process.

use std::fmt;
use tokio::io::AsyncWriteExt;

/// Control commands that can be sent to the training process
#[derive(Debug, Clone)]
pub enum ControlCommand {
    /// Pause training
    Pause,
    /// Resume training
    Resume,
    /// Save a checkpoint immediately
    SaveCheckpoint,
    /// Stop training gracefully
    Stop,
    /// Set the sample display mode
    SetSampleDisplay(SampleDisplayMode),
}

/// How to display generated samples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleDisplayMode {
    /// Show best and worst by reward
    #[default]
    BestWorst,
    /// Show random sample
    Random,
    /// Show all samples
    All,
}

impl fmt::Display for SampleDisplayMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BestWorst => write!(f, "best_worst"),
            Self::Random => write!(f, "random"),
            Self::All => write!(f, "all"),
        }
    }
}

impl SampleDisplayMode {
    /// Cycle to the next display mode
    pub fn next(self) -> Self {
        match self {
            Self::BestWorst => Self::Random,
            Self::Random => Self::All,
            Self::All => Self::BestWorst,
        }
    }

    /// Get display name for UI
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::BestWorst => "Best/Worst",
            Self::Random => "Random",
            Self::All => "All",
        }
    }
}

impl ControlCommand {
    /// Convert command to line format for stdin
    pub fn to_line(&self) -> String {
        match self {
            Self::Pause => "PAUSE".to_string(),
            Self::Resume => "RESUME".to_string(),
            Self::SaveCheckpoint => "SAVE_CHECKPOINT".to_string(),
            Self::Stop => "STOP".to_string(),
            Self::SetSampleDisplay(mode) => format!("SET sample_display={mode}"),
        }
    }
}

/// Send a control command to the training process
///
/// Has a 100ms timeout to avoid blocking if the child process is hung or dead.
pub async fn send_command(
    stdin: &mut tokio::process::ChildStdin,
    cmd: ControlCommand,
) -> std::io::Result<()> {
    use tokio::time::{Duration, timeout};

    let line = format!("{}\n", cmd.to_line());

    // Timeout after 100ms - don't block forever if child is hung
    match timeout(Duration::from_millis(100), async {
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            // Timeout - child probably dead or hung, that's fine
            Ok(())
        }
    }
}
