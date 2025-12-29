//! Output Store - Persistence layer for training outputs
//!
//! This module provides a database-backed storage system for recording
//! all model outputs during GRPO training. It supports:
//!
//! - Local SQLite file storage
//! - Remote Turso cloud database
//! - Embedded replica mode (local cache + remote sync)
//!
//! ## Usage
//!
//! ```typescript
//! // Create local store
//! const store = await OutputStore.local('./training_outputs.db');
//!
//! // Or with Turso cloud sync
//! const store = await OutputStore.embeddedReplica(
//!     './local_cache.db',
//!     'libsql://your-db.turso.io',
//!     'your-token',
//!     60 // sync interval in seconds
//! );
//!
//! // Start a training run
//! const runId = await store.startRun('qwen3-0.6b', './models/qwen3', JSON.stringify(config));
//!
//! // Record steps during training (called automatically by GRPOTrainer)
//! await store.recordStepFromOutputs(step, metrics, outputsJson, rewards, groupSize);
//!
//! // End the run
//! await store.endRun('completed');
//!
//! // Query recorded data
//! const runs = await store.listRuns();
//! const topGens = await store.getGenerationsByReward(runId, 10, 10);
//! const stats = await store.getRewardStats(runId);
//! await store.exportJsonl(runId, './analysis.jsonl');
//! ```

mod reader;
mod schema;
mod store;
mod types;
mod writer;

pub use store::OutputStore;
pub use types::*;
