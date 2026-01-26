//! Output Store - Persistence layer for training outputs
//!
//! This module provides a thin NAPI wrapper around mlx-db for storing and querying
//! training outputs. All database logic is delegated to mlx-db.
//!
//! ## Usage
//!
//! ```typescript
//! // Create local store
//! const store = await OutputStore.local('./training_outputs.db');
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
mod store;
mod types;
mod writer;

pub use store::{OutputStore, OutputStoreConfig};
pub use types::*;
