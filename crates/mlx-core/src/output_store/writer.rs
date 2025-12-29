//! Writer operations for the output store
//!
//! Handles inserting training steps, generations, and tool calls.

use libsql::Connection;
use napi::bindgen_prelude::*;

use super::types::*;
use crate::grpo::engine::EngineStepMetrics;
use crate::tools::RewardOutput;

/// Record a complete training step with all generations
///
/// Uses a transaction to ensure atomicity - all inserts succeed or none do.
pub async fn record_step(
    conn: &Connection,
    step: StepRecord,
    generations: Vec<GenerationRecord>,
    tool_calls: Vec<Vec<ToolCallRecord>>,
) -> Result<i64> {
    let created_at = chrono_now_ms();

    // Start transaction for atomicity
    let tx = conn.transaction().await.map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to begin transaction: {}", e),
        )
    })?;

    // Insert step record and get ID atomically using RETURNING clause
    let step_id = tx
        .query(
            r#"INSERT INTO training_steps
               (run_id, step, epoch, loss, mean_reward, std_reward, mean_advantage,
                total_tokens, generation_time_ms, training_time_ms, gradients_applied, created_at)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
               RETURNING id"#,
            libsql::params![
                step.run_id.clone(),
                step.step,
                step.epoch,
                step.loss,
                step.mean_reward,
                step.std_reward,
                step.mean_advantage,
                step.total_tokens,
                step.generation_time_ms,
                step.training_time_ms,
                step.gradients_applied as i32,
                created_at
            ],
        )
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to insert step: {}", e),
            )
        })?
        .next()
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to read step id: {}", e),
            )
        })?
        .ok_or_else(|| Error::new(Status::GenericFailure, "No step id returned"))?
        .get::<i64>(0)
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to parse step id: {}", e),
            )
        })?;

    // Insert generations
    for (idx, generation) in generations.iter().enumerate() {
        // Insert generation and get ID atomically using RETURNING clause
        let gen_id = tx
            .query(
                r#"INSERT INTO generations
                   (run_id, step_id, batch_index, group_index, prompt, expected_answer,
                    completion_text, completion_raw, thinking, num_tokens, finish_reason, reward, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                   RETURNING id"#,
                libsql::params![
                    step.run_id.clone(),
                    step_id,
                    generation.batch_index,
                    generation.group_index,
                    generation.prompt.clone(),
                    generation.expected_answer.clone(),
                    generation.completion_text.clone(),
                    generation.completion_raw.clone(),
                    generation.thinking.clone(),
                    generation.num_tokens,
                    generation.finish_reason.clone(),
                    generation.reward,
                    created_at
                ],
            )
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to insert generation: {}", e)))?
            .next()
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to read gen id: {}", e)))?
            .ok_or_else(|| Error::new(Status::GenericFailure, "No gen id returned"))?
            .get::<i64>(0)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to parse gen id: {}", e)))?;

        // Insert tool calls for this generation
        if idx < tool_calls.len() {
            for tc in &tool_calls[idx] {
                tx.execute(
                    r#"INSERT INTO tool_calls
                       (generation_id, call_index, status, tool_name, arguments, raw_content, error_message)
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
                    libsql::params![
                        gen_id,
                        tc.call_index,
                        tc.status.clone(),
                        tc.tool_name.clone(),
                        tc.arguments.clone(),
                        tc.raw_content.clone(),
                        tc.error_message.clone()
                    ],
                )
                .await
                .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to insert tool call: {}", e)))?;
            }
        }
    }

    // Commit transaction - all inserts succeeded
    tx.commit().await.map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to commit transaction: {}", e),
        )
    })?;

    Ok(step_id)
}

/// Record step from RewardOutput JSON (direct integration)
pub async fn record_step_from_outputs(
    conn: &Connection,
    run_id: &str,
    step: i64,
    metrics: EngineStepMetrics,
    outputs_json: &str,
    rewards: &[f64],
    group_size: i64,
) -> Result<i64> {
    // Validate group_size to prevent division by zero
    if group_size <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("group_size must be positive, got {}", group_size),
        ));
    }

    // Parse RewardOutput JSON
    let outputs: Vec<RewardOutput> = serde_json::from_str(outputs_json).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to parse outputs JSON: {}", e),
        )
    })?;

    // Build step record
    let step_record = StepRecord {
        run_id: run_id.to_string(),
        step,
        epoch: None,
        loss: metrics.loss,
        mean_reward: metrics.mean_reward,
        std_reward: metrics.std_reward,
        mean_advantage: Some(metrics.mean_advantage),
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
            batch_index,
            group_index,
            prompt: output.prompt.clone(),
            expected_answer: output.expected_answer.clone(),
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

    record_step(conn, step_record, generations, all_tool_calls).await
}

/// Get current timestamp in milliseconds
fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is set before UNIX_EPOCH")
        .as_millis() as i64
}
