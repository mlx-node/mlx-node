//! Writer operations for the output store
//!
//! Delegates to mlx-db for the actual database operations.
//! This module provides NAPI-compatible wrappers that convert between
//! the NAPI types (in grpo/engine.rs) and the plain Rust types (in mlx-db).

use napi::bindgen_prelude::*;
use sqlx::SqlitePool;

use crate::grpo::engine::EngineStepMetrics;

/// Record step from RewardOutput JSON (direct integration)
///
/// Converts NAPI EngineStepMetrics to mlx-db type and delegates to mlx_db::record_step_from_outputs.
pub async fn record_step_from_outputs(
    pool: &SqlitePool,
    run_id: &str,
    step: i64,
    metrics: EngineStepMetrics,
    outputs_json: &str,
    rewards: &[f64],
    group_size: i64,
) -> Result<i64> {
    let db_metrics = mlx_db::EngineStepMetrics::from(&metrics);

    mlx_db::record_step_from_outputs(
        pool,
        run_id,
        step,
        db_metrics,
        outputs_json,
        rewards,
        group_size,
    )
    .await
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
}

/// Record a complete training step with all generations and tool calls
///
/// Delegates to mlx_db::record_step.
pub async fn record_step(
    pool: &SqlitePool,
    step: mlx_db::StepRecord,
    generations: Vec<mlx_db::GenerationRecord>,
    tool_calls: Vec<Vec<mlx_db::ToolCallRecord>>,
) -> Result<i64> {
    mlx_db::record_step(pool, step, generations, tool_calls)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
}

/// Delete all steps after a given step number (for resume cleanup)
///
/// Delegates to mlx_db::delete_steps_after.
pub async fn delete_steps_after(pool: &SqlitePool, run_id: &str, after_step: i64) -> Result<i64> {
    mlx_db::delete_steps_after(pool, run_id, after_step)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
}

/// Delete all records after a given step (for checkpoint resume)
///
/// Cascades through: training_steps → generations → tool_calls, and logs.
/// Use this when resuming from checkpoint to ensure clean database state.
///
/// Delegates to mlx_db::delete_all_after_step.
pub async fn delete_all_after_step(
    pool: &SqlitePool,
    run_id: &str,
    after_step: i64,
) -> Result<super::types::CleanupStats> {
    let stats = mlx_db::delete_all_after_step(pool, run_id, after_step)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(stats.into())
}
