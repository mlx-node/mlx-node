//! Reader operations for the output store
//!
//! Thin NAPI wrappers around mlx_db::reader functions.
//! Converts mlx_db types to NAPI-compatible types.

use napi::bindgen_prelude::*;
use sqlx::SqlitePool;

use mlx_db::reader as db_reader;

use super::types::{
    GenerationWithToolCalls, RewardStats, RunAggregates, StepMetricSummary, StepSummary,
    TrainingRunRecord,
};

/// List all training runs
pub async fn list_runs(
    pool: &SqlitePool,
    limit: Option<i64>,
    status: Option<String>,
) -> Result<Vec<TrainingRunRecord>> {
    let runs = db_reader::list_runs(pool, limit, status.as_deref())
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(runs.into_iter().map(Into::into).collect())
}

/// Get a specific run by ID
pub async fn get_run(pool: &SqlitePool, run_id: &str) -> Result<Option<TrainingRunRecord>> {
    let run = db_reader::get_run(pool, run_id)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(run.map(Into::into))
}

/// Find a run by name
pub async fn find_run_by_name(pool: &SqlitePool, name: &str) -> Result<Option<TrainingRunRecord>> {
    let run = db_reader::find_run_by_name(pool, name)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(run.map(Into::into))
}

/// Get step summaries for a run
pub async fn get_step_summaries(
    pool: &SqlitePool,
    run_id: &str,
    start_step: Option<i64>,
    end_step: Option<i64>,
) -> Result<Vec<StepSummary>> {
    let summaries = db_reader::get_step_summaries(pool, run_id, start_step, end_step)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(summaries.into_iter().map(Into::into).collect())
}

/// Get all generations for a step
pub async fn get_generations(
    pool: &SqlitePool,
    run_id: &str,
    step: i64,
) -> Result<Vec<GenerationWithToolCalls>> {
    let gens = db_reader::get_generations(pool, run_id, step)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(gens.into_iter().map(Into::into).collect())
}

/// Get generations containing tool calls
pub async fn get_generations_with_tool_calls(
    pool: &SqlitePool,
    run_id: &str,
    tool_name: Option<String>,
    status: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let gens = db_reader::get_generations_with_tool_calls(
        pool,
        run_id,
        tool_name.as_deref(),
        status.as_deref(),
        limit,
    )
    .await
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(gens.into_iter().map(Into::into).collect())
}

/// Search generations by text content
pub async fn search_generations(
    pool: &SqlitePool,
    run_id: &str,
    query: &str,
    search_in: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let gens = db_reader::search_generations(pool, run_id, query, search_in.as_deref(), limit)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(gens.into_iter().map(Into::into).collect())
}

/// Get reward distribution statistics
pub async fn get_reward_stats(
    pool: &SqlitePool,
    run_id: &str,
    step_range: Option<Vec<i64>>,
) -> Result<RewardStats> {
    let range_slice = step_range.as_deref();
    let stats = db_reader::get_reward_stats(pool, run_id, range_slice)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(stats.into())
}

// =============================================================================
// TUI Resume State Functions
// =============================================================================

/// Get recent step metrics for sparkline restoration
pub async fn get_recent_step_metrics(
    pool: &SqlitePool,
    run_id: &str,
    limit: i64,
) -> Result<Vec<StepMetricSummary>> {
    let metrics = db_reader::get_recent_step_metrics(pool, run_id, limit)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(metrics.into_iter().map(Into::into).collect())
}

/// Get aggregate statistics for a training run
pub async fn get_run_aggregates(pool: &SqlitePool, run_id: &str) -> Result<RunAggregates> {
    let aggregates = db_reader::get_run_aggregates(pool, run_id)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(aggregates.into())
}
