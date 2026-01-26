//! OutputStore - Main storage interface for training outputs
//!
//! Thin NAPI wrapper around mlx_db for persistence.

use std::sync::Arc;

use mlx_db::schema::init_schema;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::reader;
use super::types::{
    CleanupStats, GenerationRecord, GenerationWithToolCalls, RewardStats, RunAggregates,
    StepMetricSummary, StepRecord, StepSummary, ToolCallRecord, TrainingRunRecord,
};
use super::writer;
use crate::grpo::engine::EngineStepMetrics;

/// Configuration for creating an OutputStore connection
#[napi(object)]
#[derive(Clone)]
pub struct OutputStoreConfig {
    /// Local SQLite file path (e.g., "training_outputs.db")
    pub local_path: String,
}

/// OutputStore - Persistence layer for training outputs
///
/// Stores all model outputs during GRPO training for debugging and research.
/// Supports local SQLite files.
#[napi]
pub struct OutputStore {
    pool: SqlitePool,
    config: OutputStoreConfig,
    current_run_id: Arc<RwLock<Option<String>>>,
}

#[napi]
impl OutputStore {
    /// Create a new output store with local SQLite file
    #[napi(factory)]
    pub async fn local(path: String) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", path);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to open local db: {}", e),
                )
            })?;

        init_schema(&pool)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(Self {
            pool,
            config: OutputStoreConfig { local_path: path },
            current_run_id: Arc::new(RwLock::new(None)),
        })
    }

    /// Create from config object
    #[napi(factory)]
    pub async fn from_config(config: OutputStoreConfig) -> Result<Self> {
        Self::local(config.local_path.clone()).await
    }

    // === Training Run Management ===

    /// Start a new training run
    #[napi]
    pub async fn start_run(
        &self,
        model_name: String,
        model_path: Option<String>,
        config: String,
    ) -> Result<String> {
        self.start_run_with_name(None, model_name, model_path, config)
            .await
    }

    /// Start a new training run with a name
    #[napi]
    pub async fn start_run_with_name(
        &self,
        name: Option<String>,
        model_name: String,
        model_path: Option<String>,
        config: String,
    ) -> Result<String> {
        let run_id = Uuid::new_v4().to_string();
        let started_at = chrono_now_ms();

        sqlx::query(
            "INSERT INTO training_runs (id, name, model_name, model_path, config, started_at, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
        )
        .bind(&run_id)
        .bind(&name)
        .bind(&model_name)
        .bind(&model_path)
        .bind(&config)
        .bind(started_at)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to insert run: {}", e)))?;

        let mut current = self.current_run_id.write().await;
        *current = Some(run_id.clone());

        Ok(run_id)
    }

    /// End the current training run
    #[napi]
    pub async fn end_run(&self, status: String) -> Result<()> {
        let run_id = {
            let current = self.current_run_id.read().await;
            current
                .clone()
                .ok_or_else(|| Error::new(Status::GenericFailure, "No active training run"))?
        };

        let ended_at = chrono_now_ms();

        sqlx::query("UPDATE training_runs SET ended_at = ?, status = ? WHERE id = ?")
            .bind(ended_at)
            .bind(&status)
            .bind(&run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to update run: {}", e),
                )
            })?;

        // Count steps
        let total_steps: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM training_steps WHERE run_id = ?")
                .bind(&run_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to count steps: {}", e),
                    )
                })?;

        sqlx::query("UPDATE training_runs SET total_steps = ? WHERE id = ?")
            .bind(total_steps)
            .bind(&run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to update step count: {}", e),
                )
            })?;

        let mut current = self.current_run_id.write().await;
        *current = None;

        Ok(())
    }

    /// Get current run ID
    #[napi]
    pub async fn current_run_id(&self) -> Option<String> {
        let current = self.current_run_id.read().await;
        current.clone()
    }

    /// Find a run by name
    #[napi]
    pub async fn find_run_by_name(&self, name: String) -> Result<Option<TrainingRunRecord>> {
        reader::find_run_by_name(&self.pool, &name).await
    }

    /// Resume an existing run (sets status to running and makes it current)
    #[napi]
    pub async fn resume_run(&self, run_id: String) -> Result<()> {
        sqlx::query("UPDATE training_runs SET status = 'running', ended_at = NULL WHERE id = ?")
            .bind(&run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to resume run: {}", e),
                )
            })?;

        let mut current = self.current_run_id.write().await;
        *current = Some(run_id);

        Ok(())
    }

    /// Delete all steps after a given step number (for resume cleanup)
    #[napi]
    pub async fn delete_steps_after(&self, run_id: String, after_step: i64) -> Result<i64> {
        writer::delete_steps_after(&self.pool, &run_id, after_step).await
    }

    /// Delete all records after a given step (for checkpoint resume)
    ///
    /// Cascades through: training_steps → generations → tool_calls, and logs.
    /// Use this when resuming from checkpoint to ensure clean database state.
    #[napi]
    pub async fn delete_all_after_step(
        &self,
        run_id: String,
        after_step: i64,
    ) -> Result<CleanupStats> {
        writer::delete_all_after_step(&self.pool, &run_id, after_step).await
    }

    // === TUI Resume State Methods ===

    /// Get recent step metrics for TUI sparkline restoration
    ///
    /// Returns metrics ordered by step (oldest first) for easy insertion into VecDeque.
    #[napi]
    pub async fn get_recent_step_metrics(
        &self,
        run_id: String,
        limit: i64,
    ) -> Result<Vec<StepMetricSummary>> {
        reader::get_recent_step_metrics(&self.pool, &run_id, limit).await
    }

    /// Get aggregate statistics for a training run
    ///
    /// Returns pre-computed aggregates for restoring TUI state on resume.
    #[napi]
    pub async fn get_run_aggregates(&self, run_id: String) -> Result<RunAggregates> {
        reader::get_run_aggregates(&self.pool, &run_id).await
    }

    /// Get recent generations for sample panel restoration
    ///
    /// Returns generations ordered by step DESC, reward DESC (most recent high-reward first).
    #[napi]
    pub async fn get_recent_generations(
        &self,
        run_id: String,
        limit: i64,
    ) -> Result<Vec<GenerationRecord>> {
        reader::get_recent_generations(&self.pool, &run_id, limit).await
    }

    /// Get store configuration
    #[napi(getter)]
    pub fn config(&self) -> OutputStoreConfig {
        self.config.clone()
    }

    // === Recording ===

    /// Record from RewardOutput JSON (direct integration with training engine)
    #[napi]
    pub async fn record_step_from_outputs(
        &self,
        step: i64,
        metrics: EngineStepMetrics,
        outputs_json: String,
        rewards: Vec<f64>,
        group_size: i64,
    ) -> Result<i64> {
        let run_id = {
            let current = self.current_run_id.read().await;
            current
                .clone()
                .ok_or_else(|| Error::new(Status::GenericFailure, "No active training run"))?
        };

        writer::record_step_from_outputs(
            &self.pool,
            &run_id,
            step,
            metrics,
            &outputs_json,
            &rewards,
            group_size,
        )
        .await
    }

    /// Record a complete training step with all generations and tool calls
    ///
    /// Lower-level API for direct control over step recording.
    #[napi]
    pub async fn record_step(
        &self,
        step: StepRecord,
        generations: Vec<GenerationRecord>,
        tool_calls: Vec<Vec<ToolCallRecord>>,
    ) -> Result<i64> {
        // Convert NAPI types to mlx_db types
        let db_step = mlx_db::StepRecord {
            run_id: step.run_id,
            step: step.step,
            epoch: step.epoch,
            loss: step.loss,
            mean_reward: step.mean_reward,
            std_reward: step.std_reward,
            mean_advantage: step.mean_advantage,
            std_advantage: step.std_advantage,
            total_tokens: step.total_tokens,
            generation_time_ms: step.generation_time_ms,
            training_time_ms: step.training_time_ms,
            gradients_applied: step.gradients_applied,
        };

        let db_generations: Vec<mlx_db::GenerationRecord> = generations
            .into_iter()
            .map(|g| mlx_db::GenerationRecord {
                id: None,
                step: Some(db_step.step),
                batch_index: g.batch_index,
                group_index: g.group_index,
                prompt: g.prompt,
                expected_answer: g.expected_answer,
                completion_text: g.completion_text,
                completion_raw: g.completion_raw,
                thinking: g.thinking,
                num_tokens: g.num_tokens,
                finish_reason: g.finish_reason,
                reward: g.reward,
            })
            .collect();

        let db_tool_calls: Vec<Vec<mlx_db::ToolCallRecord>> = tool_calls
            .into_iter()
            .map(|tcs| {
                tcs.into_iter()
                    .map(|tc| mlx_db::ToolCallRecord {
                        id: None,
                        generation_id: None,
                        call_index: tc.call_index,
                        status: tc.status,
                        tool_name: tc.tool_name,
                        arguments: tc.arguments,
                        raw_content: tc.raw_content,
                        error_message: tc.error_message,
                    })
                    .collect()
            })
            .collect();

        writer::record_step(&self.pool, db_step, db_generations, db_tool_calls).await
    }

    /// Flush any pending writes
    #[napi]
    pub async fn flush(&self) -> Result<()> {
        Ok(())
    }

    // === Query API ===

    /// List all training runs
    #[napi]
    pub async fn list_runs(
        &self,
        limit: Option<i64>,
        status: Option<String>,
    ) -> Result<Vec<TrainingRunRecord>> {
        reader::list_runs(&self.pool, limit, status).await
    }

    /// Get a specific run
    #[napi]
    pub async fn get_run(&self, run_id: String) -> Result<Option<TrainingRunRecord>> {
        reader::get_run(&self.pool, &run_id).await
    }

    /// Get step summaries for a run
    #[napi]
    pub async fn get_step_summaries(
        &self,
        run_id: String,
        start_step: Option<i64>,
        end_step: Option<i64>,
    ) -> Result<Vec<StepSummary>> {
        reader::get_step_summaries(&self.pool, &run_id, start_step, end_step).await
    }

    /// Get all generations for a step
    #[napi]
    pub async fn get_generations(
        &self,
        run_id: String,
        step: i64,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        reader::get_generations(&self.pool, &run_id, step).await
    }

    /// Get top/bottom generations by reward
    #[napi]
    pub async fn get_generations_by_reward(
        &self,
        run_id: String,
        top_n: Option<i64>,
        bottom_n: Option<i64>,
        step_range: Option<Vec<i64>>,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        reader::get_generations_by_reward(&self.pool, &run_id, top_n, bottom_n, step_range).await
    }

    /// Get generations with specific finish reason
    #[napi]
    pub async fn get_generations_by_finish_reason(
        &self,
        run_id: String,
        finish_reason: String,
        limit: Option<i64>,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        reader::get_generations_by_finish_reason(&self.pool, &run_id, &finish_reason, limit).await
    }

    /// Get generations containing tool calls
    #[napi]
    pub async fn get_generations_with_tool_calls(
        &self,
        run_id: String,
        tool_name: Option<String>,
        status: Option<String>,
        limit: Option<i64>,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        reader::get_generations_with_tool_calls(&self.pool, &run_id, tool_name, status, limit).await
    }

    /// Search generations by text content
    #[napi]
    pub async fn search_generations(
        &self,
        run_id: String,
        query: String,
        search_in: Option<String>,
        limit: Option<i64>,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        reader::search_generations(&self.pool, &run_id, &query, search_in, limit).await
    }

    /// Get reward distribution statistics
    #[napi]
    pub async fn get_reward_stats(
        &self,
        run_id: String,
        step_range: Option<Vec<i64>>,
    ) -> Result<RewardStats> {
        reader::get_reward_stats(&self.pool, &run_id, step_range).await
    }

    /// Export to JSONL file
    #[napi]
    pub async fn export_jsonl(
        &self,
        run_id: String,
        output_path: String,
        include_tool_calls: Option<bool>,
    ) -> Result<i64> {
        reader::export_jsonl(
            &self.pool,
            &run_id,
            &output_path,
            include_tool_calls.unwrap_or(true),
        )
        .await
    }

    /// Execute raw SQL query (for advanced users)
    #[napi]
    pub async fn query_raw(&self, sql: String) -> Result<String> {
        reader::query_raw(&self.pool, &sql).await
    }
}

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is set before UNIX_EPOCH")
        .as_millis() as i64
}
