#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

//! MLX-DB: Shared database operations for MLX-Node
//!
//! This crate provides SQLite database operations using sqlx for storing and querying
//! training outputs. It's shared between:
//! - `mlx-core`: NAPI bindings for Node.js (async API)
//! - `mlx-tui`: Native Rust TUI (sync API via `SyncDbReader`)
//!
//! ## Features
//!
//! - Training run management with named runs for resume support
//! - Step and generation recording with tool call tracking
//! - Flexible filtering and querying (by reward, step, finish reason, etc.)
//! - Export to JSONL for external analysis
//! - Schema migration for existing databases
//!
//! ## Usage
//!
//! ### Async (for mlx-core/Node.js)
//!
//! Reader and writer functions take a `&SqlitePool` opened by the caller.
//!
//! ### Sync (for mlx-tui)
//!
//! ```ignore
//! use mlx_db::{SyncDbReader, reader};
//!
//! let db = SyncDbReader::open("outputs.db")?;
//! let runs = db.block_on(reader::list_runs(db.pool(), None, None))?;
//! ```

pub mod connection;
pub mod error;
pub mod reader;
pub mod response_store;
pub mod schema;
pub mod types;
pub mod writer;

// Re-export main types for convenience
pub use connection::SyncDbReader;
pub use error::DbError;
pub use types::*;

// Re-export commonly used reader functions
pub use reader::{
    get_generations_filtered, get_logs, get_recent_generations, get_recent_step_metrics, get_run,
    get_run_aggregates, get_step_summaries, list_runs,
};

// Re-export writer functions
pub use writer::{
    CleanupStats, delete_all_after_step, delete_steps_after, record_step, record_step_from_outputs,
};

// Re-export training types
pub use types::EngineStepMetrics;
