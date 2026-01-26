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
//! ```ignore
//! use mlx_db::{AsyncDb, reader, writer};
//!
//! let db = AsyncDb::local("outputs.db").await?;
//! let runs = reader::list_runs(db.pool(), None, None).await?;
//! ```
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
pub mod schema;
pub mod types;
pub mod writer;

// Re-export main types for convenience
pub use connection::{AsyncDb, SyncDbReader};
pub use error::DbError;
pub use types::*;

// Re-export commonly used reader functions
pub use reader::{
    get_generations_filtered, get_log_count, get_logs, get_recent_generations,
    get_recent_step_metrics, get_run, get_run_aggregates, get_step_summaries,
    get_tool_calls_for_generation, list_runs,
};

// Re-export writer functions
pub use writer::{
    CleanupStats, delete_all_after_step, delete_steps_after, increment_run_steps, record_step,
    record_step_from_outputs, write_log,
};

// Re-export training types
pub use types::EngineStepMetrics;
