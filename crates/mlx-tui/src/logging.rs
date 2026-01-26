//! Unified logging infrastructure using tracing
//!
//! Provides file logging with optional future database persistence.
//!
//! # Environment Variables
//!
//! - `MLX_TUI_DEBUG=1`: Enable debug-level logging (default: errors only)
//! - `MLX_TUI_LOG_PATH`: Custom log file path (default: `/tmp/mlx-tui-debug.log`)
//!
//! # Usage
//!
//! ```ignore
//! // Initialize at startup (before any logging)
//! // Hold the guard until program exit to ensure logs flush
//! let _log_guard = logging::init();
//!
//! // Use tracing macros
//! tracing::debug!("Debug message");
//! tracing::info!("Info message");
//! tracing::warn!("Warning");
//! tracing::error!("Error");
//! ```

use std::path::PathBuf;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize the unified logging system.
///
/// Returns a guard that must be held for the lifetime of the app to ensure logs flush.
/// Drop this guard before the tokio runtime shuts down to avoid panics.
pub fn init() -> WorkerGuard {
    let debug_enabled = std::env::var("MLX_TUI_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);

    let log_path = std::env::var("MLX_TUI_LOG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/mlx-tui-debug.log"));

    // Create file appender (non-blocking for async compatibility)
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("Failed to open log file");

    let (non_blocking, guard) = tracing_appender::non_blocking(file);

    // Build env filter based on debug flag
    let filter = if debug_enabled {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("mlx_tui=debug,ui=debug"))
    } else {
        // Only log errors when debug is disabled
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("mlx_tui=error,ui=error"))
    };

    // Configure subscriber with file output
    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(true)
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    guard
}

// Note: Database persistence for logs is handled separately through the logs table
// in mlx-db. The schema and writer functions are available but the actual
// persistence from the tracing layer requires a more complex async/sync bridge.
// For now, file logging provides crash recovery and UI logs are persisted via
// the add_log() function which emits to tracing.
