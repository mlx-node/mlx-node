//! Writer operations for the output store
//!
//! Handles inserting and updating training runs, steps, generations, and tool calls.

use sqlx::SqlitePool;
use uuid::Uuid;

use crate::error::DbError;
use crate::types::{EngineStepMetrics, GenerationRecord, StepRecord, ToolCallRecord};

/// Start a new training run
pub async fn start_run(
    pool: &SqlitePool,
    name: Option<&str>,
    model_name: &str,
    model_path: Option<&str>,
    config: &str,
) -> Result<String, DbError> {
    let run_id = Uuid::new_v4().to_string();
    let started_at = chrono_now_ms();

    sqlx::query(
        "INSERT INTO training_runs (id, name, model_name, model_path, config, started_at, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
    )
    .bind(&run_id)
    .bind(name)
    .bind(model_name)
    .bind(model_path)
    .bind(config)
    .bind(started_at)
    .execute(pool)
    .await
    .map_err(|e| DbError::Write(format!("Failed to start run: {}", e)))?;

    Ok(run_id)
}

/// End a training run
pub async fn end_run(pool: &SqlitePool, run_id: &str, status: &str) -> Result<(), DbError> {
    let ended_at = chrono_now_ms();

    sqlx::query("UPDATE training_runs SET ended_at = ?, status = ? WHERE id = ?")
        .bind(ended_at)
        .bind(status)
        .bind(run_id)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to end run: {}", e)))?;

    Ok(())
}

/// Resume an existing run (set status back to running)
pub async fn resume_run(pool: &SqlitePool, run_id: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE training_runs SET status = 'running', ended_at = NULL WHERE id = ?")
        .bind(run_id)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to resume run: {}", e)))?;

    Ok(())
}

/// Update run's total steps
pub async fn update_run_steps(pool: &SqlitePool, run_id: &str, steps: i64) -> Result<(), DbError> {
    sqlx::query("UPDATE training_runs SET total_steps = ? WHERE id = ?")
        .bind(steps)
        .bind(run_id)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to update steps: {}", e)))?;

    Ok(())
}

/// Record a complete training step with all generations
///
/// Uses a transaction to ensure atomicity - all inserts succeed or none do.
pub async fn record_step(
    pool: &SqlitePool,
    step: StepRecord,
    generations: Vec<GenerationRecord>,
    tool_calls: Vec<Vec<ToolCallRecord>>,
) -> Result<i64, DbError> {
    let created_at = chrono_now_ms();

    // Start transaction for atomicity
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| DbError::Transaction(format!("Failed to begin transaction: {}", e)))?;

    // Insert step record and get ID atomically using RETURNING clause
    let step_id: i64 = sqlx::query_scalar(
        r#"INSERT INTO training_steps
           (run_id, step, epoch, loss, mean_reward, std_reward, mean_advantage, std_advantage,
            total_tokens, generation_time_ms, training_time_ms, gradients_applied, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id"#,
    )
    .bind(&step.run_id)
    .bind(step.step)
    .bind(step.epoch)
    .bind(step.loss)
    .bind(step.mean_reward)
    .bind(step.std_reward)
    .bind(step.mean_advantage)
    .bind(step.std_advantage)
    .bind(step.total_tokens)
    .bind(step.generation_time_ms)
    .bind(step.training_time_ms)
    .bind(step.gradients_applied as i32)
    .bind(created_at)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| DbError::Write(format!("Failed to insert step: {}", e)))?;

    // Insert generations
    for (idx, generation) in generations.iter().enumerate() {
        // Insert generation and get ID atomically using RETURNING clause
        let gen_id: i64 = sqlx::query_scalar(
            r#"INSERT INTO generations
               (run_id, step_id, batch_index, group_index, prompt, expected_answer,
                completion_text, completion_raw, thinking, num_tokens, finish_reason, reward, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               RETURNING id"#,
        )
        .bind(&step.run_id)
        .bind(step_id)
        .bind(generation.batch_index)
        .bind(generation.group_index)
        .bind(&generation.prompt)
        .bind(&generation.expected_answer)
        .bind(&generation.completion_text)
        .bind(&generation.completion_raw)
        .bind(&generation.thinking)
        .bind(generation.num_tokens)
        .bind(&generation.finish_reason)
        .bind(generation.reward)
        .bind(created_at)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| DbError::Write(format!("Failed to insert generation: {}", e)))?;

        // Insert tool calls for this generation
        if idx < tool_calls.len() {
            for tc in &tool_calls[idx] {
                sqlx::query(
                    r#"INSERT INTO tool_calls
                       (generation_id, call_index, status, tool_name, arguments, raw_content, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?)"#,
                )
                .bind(gen_id)
                .bind(tc.call_index)
                .bind(&tc.status)
                .bind(&tc.tool_name)
                .bind(&tc.arguments)
                .bind(&tc.raw_content)
                .bind(&tc.error_message)
                .execute(&mut *tx)
                .await
                .map_err(|e| DbError::Write(format!("Failed to insert tool call: {}", e)))?;
            }
        }
    }

    // Commit transaction - all inserts succeeded
    tx.commit()
        .await
        .map_err(|e| DbError::Transaction(format!("Failed to commit transaction: {}", e)))?;

    Ok(step_id)
}

/// Write a log entry to the database
pub async fn write_log(
    pool: &SqlitePool,
    run_id: Option<&str>,
    level: &str,
    target: &str,
    message: &str,
    file: Option<&str>,
    line: Option<u32>,
) -> Result<(), DbError> {
    let created_at = chrono_now_ms();

    sqlx::query(
        "INSERT INTO logs (run_id, level, target, message, file, line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(run_id)
    .bind(level)
    .bind(target)
    .bind(message)
    .bind(file)
    .bind(line.map(|l| l as i64))
    .bind(created_at)
    .execute(pool)
    .await
    .map_err(|e| DbError::Write(format!("Failed to write log: {}", e)))?;

    Ok(())
}

/// Increment the total_steps counter for a training run
///
/// Called after each step is recorded to keep total_steps accurate even on crash.
pub async fn increment_run_steps(pool: &SqlitePool, run_id: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE training_runs SET total_steps = total_steps + 1 WHERE id = ?")
        .bind(run_id)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to increment steps: {}", e)))?;

    Ok(())
}

/// Get current timestamp in milliseconds
fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is set before UNIX_EPOCH")
        .as_millis() as i64
}

/// Record step from RewardOutput JSON (direct integration)
///
/// Parses the RewardOutput JSON array and builds generation/tool_call records,
/// then calls record_step() for atomic persistence.
pub async fn record_step_from_outputs(
    pool: &SqlitePool,
    run_id: &str,
    step: i64,
    metrics: EngineStepMetrics,
    outputs_json: &str,
    rewards: &[f64],
    group_size: i64,
) -> Result<i64, DbError> {
    // Validate group_size to prevent division by zero
    if group_size <= 0 {
        return Err(DbError::InvalidArg(format!(
            "group_size must be positive, got {}",
            group_size
        )));
    }

    // Parse RewardOutput JSON
    let outputs: Vec<crate::types::RewardOutput> = serde_json::from_str(outputs_json)
        .map_err(|e| DbError::Serialization(format!("Failed to parse outputs JSON: {}", e)))?;

    // Build step record
    let step_record = StepRecord {
        run_id: run_id.to_string(),
        step,
        epoch: None,
        loss: metrics.loss,
        mean_reward: metrics.mean_reward,
        std_reward: metrics.std_reward,
        mean_advantage: Some(metrics.mean_advantage),
        std_advantage: metrics.std_advantage,
        total_tokens: Some(metrics.total_tokens as i64),
        generation_time_ms: Some(metrics.generation_time_ms),
        training_time_ms: Some(metrics.training_time_ms),
        gradients_applied: metrics.gradients_applied,
    };

    // Build generation records and tool calls
    let mut generations = Vec::with_capacity(outputs.len());
    let mut all_tool_calls = Vec::with_capacity(outputs.len());

    for (idx, output) in outputs.iter().enumerate() {
        let batch_index = (idx / group_size as usize) as i64;
        let group_index = (idx % group_size as usize) as i64;

        let generation = GenerationRecord {
            id: None,
            step: Some(step),
            batch_index,
            group_index,
            prompt: output.prompt.clone(),
            expected_answer: None,
            completion_text: output.completion.text.clone(),
            completion_raw: output.completion.raw_text.clone(),
            thinking: output.completion.thinking.clone(),
            num_tokens: output.completion.num_tokens as i64,
            finish_reason: output.completion.finish_reason.clone(),
            reward: if idx < rewards.len() {
                rewards[idx]
            } else {
                0.0
            },
        };
        generations.push(generation);

        // Convert tool calls
        let tool_calls: Vec<ToolCallRecord> = output
            .completion
            .tool_calls
            .iter()
            .enumerate()
            .map(|(call_idx, tc)| ToolCallRecord {
                id: None,
                generation_id: None,
                call_index: call_idx as i64,
                status: tc.status.clone(),
                tool_name: Some(tc.name.clone()),
                arguments: Some(serde_json::to_string(&tc.arguments).unwrap_or_default()),
                raw_content: tc.raw_content.clone(),
                error_message: tc.error.clone(),
            })
            .collect();
        all_tool_calls.push(tool_calls);
    }

    record_step(pool, step_record, generations, all_tool_calls).await
}

/// Statistics about cleanup operations
#[derive(Debug, Clone, Default)]
pub struct CleanupStats {
    /// Number of training steps deleted
    pub steps_deleted: u64,
    /// Number of generations deleted
    pub generations_deleted: u64,
    /// Number of tool calls deleted
    pub tool_calls_deleted: u64,
    /// Number of logs deleted
    pub logs_deleted: u64,
}

/// Delete all steps after a given step number (for resume cleanup)
///
/// This is used when resuming from a checkpoint to ensure database state
/// matches the checkpoint state. Deletes:
/// - All tool_calls for affected generations
/// - All generations for steps > after_step
/// - All training_steps where step > after_step
///
/// Note: sqlx supports IN (SELECT ...) subqueries, so we can simplify the query.
pub async fn delete_steps_after(
    pool: &SqlitePool,
    run_id: &str,
    after_step: i64,
) -> Result<i64, DbError> {
    // Count steps to delete first
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM training_steps WHERE run_id = ? AND step > ?")
            .bind(run_id)
            .bind(after_step)
            .fetch_one(pool)
            .await
            .map_err(|e| DbError::Query(format!("Failed to count steps: {}", e)))?;

    if count == 0 {
        return Ok(0);
    }

    // Delete tool_calls for affected generations using subquery
    sqlx::query(
        "DELETE FROM tool_calls WHERE generation_id IN (
            SELECT g.id FROM generations g
            JOIN training_steps ts ON g.step_id = ts.id
            WHERE ts.run_id = ? AND ts.step > ?)",
    )
    .bind(run_id)
    .bind(after_step)
    .execute(pool)
    .await
    .map_err(|e| DbError::Write(format!("Failed to delete tool_calls: {}", e)))?;

    // Delete generations for affected steps using subquery
    sqlx::query(
        "DELETE FROM generations WHERE step_id IN (
            SELECT id FROM training_steps WHERE run_id = ? AND step > ?)",
    )
    .bind(run_id)
    .bind(after_step)
    .execute(pool)
    .await
    .map_err(|e| DbError::Write(format!("Failed to delete generations: {}", e)))?;

    // Delete the steps themselves
    sqlx::query("DELETE FROM training_steps WHERE run_id = ? AND step > ?")
        .bind(run_id)
        .bind(after_step)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to delete steps: {}", e)))?;

    Ok(count)
}

/// Delete all records after a given step (for checkpoint resume)
///
/// This is a more comprehensive cleanup that also removes orphaned logs.
/// Cascades through: training_steps → generations → tool_calls, and logs.
///
/// Use this when resuming from checkpoint to ensure clean database state
/// without stale data from previous crashed runs.
pub async fn delete_all_after_step(
    pool: &SqlitePool,
    run_id: &str,
    after_step: i64,
) -> Result<CleanupStats, DbError> {
    let mut stats = CleanupStats::default();

    // Count steps to delete first
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM training_steps WHERE run_id = ? AND step > ?")
            .bind(run_id)
            .bind(after_step)
            .fetch_one(pool)
            .await
            .map_err(|e| DbError::Query(format!("Failed to count steps: {}", e)))?;

    if count == 0 {
        // No steps to delete, but still check for orphaned logs
        // (e.g., from a crash before step was fully recorded)
    } else {
        // 1. Delete tool_calls for affected generations
        let tool_calls_result = sqlx::query(
            "DELETE FROM tool_calls WHERE generation_id IN (
                SELECT g.id FROM generations g
                JOIN training_steps ts ON g.step_id = ts.id
                WHERE ts.run_id = ? AND ts.step > ?)",
        )
        .bind(run_id)
        .bind(after_step)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to delete tool_calls: {}", e)))?;
        stats.tool_calls_deleted = tool_calls_result.rows_affected();

        // 2. Delete generations for affected steps
        let generations_result = sqlx::query(
            "DELETE FROM generations WHERE step_id IN (
                SELECT id FROM training_steps WHERE run_id = ? AND step > ?)",
        )
        .bind(run_id)
        .bind(after_step)
        .execute(pool)
        .await
        .map_err(|e| DbError::Write(format!("Failed to delete generations: {}", e)))?;
        stats.generations_deleted = generations_result.rows_affected();

        // 3. Delete training_steps
        let steps_result = sqlx::query("DELETE FROM training_steps WHERE run_id = ? AND step > ?")
            .bind(run_id)
            .bind(after_step)
            .execute(pool)
            .await
            .map_err(|e| DbError::Write(format!("Failed to delete steps: {}", e)))?;
        stats.steps_deleted = steps_result.rows_affected();
    }

    // 4. Delete logs after checkpoint timestamp
    // Get timestamp from checkpoint step to delete logs created after that
    let checkpoint_time: Option<i64> =
        sqlx::query_scalar("SELECT created_at FROM training_steps WHERE run_id = ? AND step = ?")
            .bind(run_id)
            .bind(after_step)
            .fetch_optional(pool)
            .await
            .map_err(|e| DbError::Query(format!("Failed to get checkpoint timestamp: {}", e)))?;

    if let Some(ts) = checkpoint_time {
        let logs_result = sqlx::query("DELETE FROM logs WHERE run_id = ? AND created_at > ?")
            .bind(run_id)
            .bind(ts)
            .execute(pool)
            .await
            .map_err(|e| DbError::Write(format!("Failed to delete logs: {}", e)))?;
        stats.logs_deleted = logs_result.rows_affected();
    }

    Ok(stats)
}
