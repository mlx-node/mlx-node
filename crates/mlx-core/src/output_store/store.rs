//! OutputStore - Main storage interface for training outputs
//!
//! Provides connection management and high-level API for recording
//! and querying training data.

use std::sync::Arc;
use std::time::Duration;

use libsql::{Builder, Connection, Database};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::reader;
use super::schema::init_schema;
use super::types::*;
use super::writer;
use crate::grpo::engine::EngineStepMetrics;

/// OutputStore - Persistence layer for training outputs
///
/// Stores all model outputs during GRPO training for debugging and research.
/// Supports local SQLite files and remote Turso cloud databases.
#[napi]
pub struct OutputStore {
    db: Arc<Database>,
    conn: Arc<RwLock<Connection>>,
    config: OutputStoreConfig,
    current_run_id: Arc<RwLock<Option<String>>>,
}

#[napi]
impl OutputStore {
    /// Create a new output store with local SQLite file
    #[napi(factory)]
    pub async fn local(path: String) -> Result<Self> {
        let db = Builder::new_local(&path).build().await.map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to open local db: {}", e),
            )
        })?;

        let conn = db
            .connect()
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to connect: {}", e)))?;

        // Initialize schema
        init_schema(&conn).await?;

        Ok(Self {
            db: Arc::new(db),
            conn: Arc::new(RwLock::new(conn)),
            config: OutputStoreConfig {
                local_path: Some(path),
                ..Default::default()
            },
            current_run_id: Arc::new(RwLock::new(None)),
        })
    }

    /// Create a new output store with remote Turso database
    #[napi(factory)]
    pub async fn remote(url: String, token: String) -> Result<Self> {
        let db = Builder::new_remote(url.clone(), token.clone())
            .build()
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to connect to remote db: {}", e),
                )
            })?;

        let conn = db
            .connect()
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to connect: {}", e)))?;

        // Initialize schema
        init_schema(&conn).await?;

        Ok(Self {
            db: Arc::new(db),
            conn: Arc::new(RwLock::new(conn)),
            config: OutputStoreConfig {
                remote_url: Some(url),
                auth_token: Some(token),
                ..Default::default()
            },
            current_run_id: Arc::new(RwLock::new(None)),
        })
    }

    /// Create embedded replica (local cache + remote sync)
    #[napi(factory)]
    pub async fn embedded_replica(
        local_path: String,
        remote_url: String,
        token: String,
        sync_interval_secs: Option<u32>,
    ) -> Result<Self> {
        let interval = sync_interval_secs.unwrap_or(60);

        let db = Builder::new_remote_replica(&local_path, remote_url.clone(), token.clone())
            .sync_interval(Duration::from_secs(interval as u64))
            .build()
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to create replica: {}", e),
                )
            })?;

        let conn = db
            .connect()
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to connect: {}", e)))?;

        // Initialize schema
        init_schema(&conn).await?;

        // Initial sync
        db.sync().await.map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Initial sync failed: {}", e),
            )
        })?;

        Ok(Self {
            db: Arc::new(db),
            conn: Arc::new(RwLock::new(conn)),
            config: OutputStoreConfig {
                local_path: Some(local_path),
                remote_url: Some(remote_url),
                auth_token: Some(token),
                sync_interval_secs: Some(interval),
                ..Default::default()
            },
            current_run_id: Arc::new(RwLock::new(None)),
        })
    }

    /// Create from config object
    #[napi(factory)]
    pub async fn from_config(config: OutputStoreConfig) -> Result<Self> {
        if let (Some(local), Some(remote), Some(token)) =
            (&config.local_path, &config.remote_url, &config.auth_token)
        {
            // Embedded replica mode
            Self::embedded_replica(
                local.clone(),
                remote.clone(),
                token.clone(),
                config.sync_interval_secs,
            )
            .await
        } else if let (Some(remote), Some(token)) = (&config.remote_url, &config.auth_token) {
            // Remote only
            Self::remote(remote.clone(), token.clone()).await
        } else if let Some(local) = &config.local_path {
            // Local only
            Self::local(local.clone()).await
        } else {
            Err(Error::new(
                Status::InvalidArg,
                "OutputStoreConfig must specify either local_path or remote_url + auth_token",
            ))
        }
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
        let run_id = Uuid::new_v4().to_string();
        let started_at = chrono_now_ms();

        let conn = self.conn.write().await;
        conn.execute(
            "INSERT INTO training_runs (id, model_name, model_path, config, started_at, status) VALUES (?1, ?2, ?3, ?4, ?5, 'running')",
            libsql::params![run_id.clone(), model_name, model_path, config, started_at],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to insert run: {}", e)))?;

        // Store current run ID
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

        let conn = self.conn.write().await;
        conn.execute(
            "UPDATE training_runs SET ended_at = ?1, status = ?2 WHERE id = ?3",
            libsql::params![ended_at, status, run_id.clone()],
        )
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to update run: {}", e),
            )
        })?;

        // Update step count
        conn.execute(
            "UPDATE training_runs SET total_steps = (SELECT COUNT(*) FROM training_steps WHERE run_id = ?1) WHERE id = ?1",
            libsql::params![run_id],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to update step count: {}", e)))?;

        // Clear current run ID
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

    /// Get store configuration
    #[napi(getter)]
    pub fn config(&self) -> OutputStoreConfig {
        self.config.clone()
    }

    // === Recording ===

    /// Record a complete training step with all generations
    #[napi]
    pub async fn record_step(
        &self,
        step: StepRecord,
        generations: Vec<GenerationRecord>,
        tool_calls: Vec<Vec<ToolCallRecord>>,
    ) -> Result<i64> {
        let conn = self.conn.write().await;
        writer::record_step(&conn, step, generations, tool_calls).await
    }

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

        let conn = self.conn.write().await;
        writer::record_step_from_outputs(
            &conn,
            &run_id,
            step,
            metrics,
            &outputs_json,
            &rewards,
            group_size,
        )
        .await
    }

    /// Flush any pending writes
    #[napi]
    pub async fn flush(&self) -> Result<()> {
        // For now, all writes are immediate
        // Future: implement batched writes
        Ok(())
    }

    /// Sync with remote (for embedded replica mode)
    #[napi]
    pub async fn sync(&self) -> Result<()> {
        self.db
            .sync()
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Sync failed: {}", e)))?;
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
        let conn = self.conn.read().await;
        reader::list_runs(&conn, limit, status).await
    }

    /// Get a specific run
    #[napi]
    pub async fn get_run(&self, run_id: String) -> Result<Option<TrainingRunRecord>> {
        let conn = self.conn.read().await;
        reader::get_run(&conn, &run_id).await
    }

    /// Get step summaries for a run
    #[napi]
    pub async fn get_step_summaries(
        &self,
        run_id: String,
        start_step: Option<i64>,
        end_step: Option<i64>,
    ) -> Result<Vec<StepSummary>> {
        let conn = self.conn.read().await;
        reader::get_step_summaries(&conn, &run_id, start_step, end_step).await
    }

    /// Get all generations for a step
    #[napi]
    pub async fn get_generations(
        &self,
        run_id: String,
        step: i64,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        let conn = self.conn.read().await;
        reader::get_generations(&conn, &run_id, step).await
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
        let conn = self.conn.read().await;
        reader::get_generations_by_reward(&conn, &run_id, top_n, bottom_n, step_range).await
    }

    /// Get generations with specific finish reason
    #[napi]
    pub async fn get_generations_by_finish_reason(
        &self,
        run_id: String,
        finish_reason: String,
        limit: Option<i64>,
    ) -> Result<Vec<GenerationWithToolCalls>> {
        let conn = self.conn.read().await;
        reader::get_generations_by_finish_reason(&conn, &run_id, &finish_reason, limit).await
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
        let conn = self.conn.read().await;
        reader::get_generations_with_tool_calls(&conn, &run_id, tool_name, status, limit).await
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
        let conn = self.conn.read().await;
        reader::search_generations(&conn, &run_id, &query, search_in, limit).await
    }

    /// Get reward distribution statistics
    #[napi]
    pub async fn get_reward_stats(
        &self,
        run_id: String,
        step_range: Option<Vec<i64>>,
    ) -> Result<RewardStats> {
        let conn = self.conn.read().await;
        reader::get_reward_stats(&conn, &run_id, step_range).await
    }

    /// Export to JSONL file
    #[napi]
    pub async fn export_jsonl(
        &self,
        run_id: String,
        output_path: String,
        include_tool_calls: Option<bool>,
    ) -> Result<i64> {
        let conn = self.conn.read().await;
        reader::export_jsonl(
            &conn,
            &run_id,
            &output_path,
            include_tool_calls.unwrap_or(true),
        )
        .await
    }

    /// Execute raw SQL query (for advanced users)
    #[napi]
    pub async fn query_raw(&self, sql: String) -> Result<String> {
        let conn = self.conn.read().await;
        reader::query_raw(&conn, &sql).await
    }
}

/// Get current timestamp in milliseconds
fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is set before UNIX_EPOCH")
        .as_millis() as i64
}
