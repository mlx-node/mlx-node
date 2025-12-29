//! Reader operations for the output store
//!
//! Handles querying training runs, steps, generations, and tool calls.

use std::fs::File;
use std::io::{BufWriter, Write};

use libsql::Connection;
use napi::bindgen_prelude::*;
use serde_json::json;

use super::types::*;

/// List all training runs
pub async fn list_runs(
    conn: &Connection,
    limit: Option<i64>,
    status: Option<String>,
) -> Result<Vec<TrainingRunRecord>> {
    let mut sql = "SELECT id, model_name, model_path, config, started_at, ended_at, total_steps, status FROM training_runs".to_string();

    if status.is_some() {
        sql.push_str(" WHERE status = ?1");
    }

    sql.push_str(" ORDER BY started_at DESC");

    if let Some(lim) = limit {
        sql.push_str(&format!(" LIMIT {}", lim));
    }

    let mut rows = if let Some(ref st) = status {
        conn.query(&sql, libsql::params![st.clone()]).await
    } else {
        conn.query(&sql, ()).await
    }
    .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut runs = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        runs.push(TrainingRunRecord {
            id: row.get::<String>(0).unwrap_or_default(),
            model_name: row.get::<String>(1).unwrap_or_default(),
            model_path: row.get::<Option<String>>(2).ok().flatten(),
            config: row.get::<String>(3).unwrap_or_default(),
            started_at: row.get::<i64>(4).unwrap_or(0),
            ended_at: row.get::<Option<i64>>(5).ok().flatten(),
            total_steps: row.get::<i64>(6).unwrap_or(0),
            status: row.get::<String>(7).unwrap_or_default(),
        });
    }

    Ok(runs)
}

/// Get a specific run
pub async fn get_run(conn: &Connection, run_id: &str) -> Result<Option<TrainingRunRecord>> {
    let mut rows = conn
        .query(
            "SELECT id, model_name, model_path, config, started_at, ended_at, total_steps, status FROM training_runs WHERE id = ?1",
            libsql::params![run_id.to_string()],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        Ok(Some(TrainingRunRecord {
            id: row.get::<String>(0).unwrap_or_default(),
            model_name: row.get::<String>(1).unwrap_or_default(),
            model_path: row.get::<Option<String>>(2).ok().flatten(),
            config: row.get::<String>(3).unwrap_or_default(),
            started_at: row.get::<i64>(4).unwrap_or(0),
            ended_at: row.get::<Option<i64>>(5).ok().flatten(),
            total_steps: row.get::<i64>(6).unwrap_or(0),
            status: row.get::<String>(7).unwrap_or_default(),
        }))
    } else {
        Ok(None)
    }
}

/// Get step summaries for a run
pub async fn get_step_summaries(
    conn: &Connection,
    run_id: &str,
    start_step: Option<i64>,
    end_step: Option<i64>,
) -> Result<Vec<StepSummary>> {
    let mut sql = r#"
        SELECT
            ts.step,
            ts.loss,
            ts.mean_reward,
            COUNT(g.id) as num_generations,
            (SELECT COUNT(*) FROM tool_calls tc WHERE tc.generation_id IN (SELECT id FROM generations WHERE step_id = ts.id)) as num_tool_calls,
            SUM(CASE WHEN g.finish_reason = 'eos' THEN 1 ELSE 0 END) as eos_count,
            SUM(CASE WHEN g.finish_reason = 'length' THEN 1 ELSE 0 END) as length_count
        FROM training_steps ts
        LEFT JOIN generations g ON g.step_id = ts.id
        WHERE ts.run_id = ?1
    "#.to_string();

    let mut param_idx = 2;
    if start_step.is_some() {
        sql.push_str(&format!(" AND ts.step >= ?{}", param_idx));
        param_idx += 1;
    }
    if end_step.is_some() {
        sql.push_str(&format!(" AND ts.step <= ?{}", param_idx));
    }

    sql.push_str(" GROUP BY ts.id ORDER BY ts.step");

    let mut rows = match (start_step, end_step) {
        (Some(s), Some(e)) => {
            conn.query(&sql, libsql::params![run_id.to_string(), s, e])
                .await
        }
        (Some(s), None) => {
            conn.query(&sql, libsql::params![run_id.to_string(), s])
                .await
        }
        (None, Some(e)) => {
            conn.query(&sql, libsql::params![run_id.to_string(), e])
                .await
        }
        (None, None) => conn.query(&sql, libsql::params![run_id.to_string()]).await,
    }
    .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut summaries = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        summaries.push(StepSummary {
            step: row.get::<i64>(0).unwrap_or(0),
            loss: row.get::<f64>(1).unwrap_or(0.0),
            mean_reward: row.get::<f64>(2).unwrap_or(0.0),
            num_generations: row.get::<i64>(3).unwrap_or(0),
            num_tool_calls: row.get::<i64>(4).unwrap_or(0),
            eos_count: row.get::<i64>(5).unwrap_or(0),
            length_count: row.get::<i64>(6).unwrap_or(0),
        });
    }

    Ok(summaries)
}

/// Get all generations for a step
pub async fn get_generations(
    conn: &Connection,
    run_id: &str,
    step: i64,
) -> Result<Vec<GenerationWithToolCalls>> {
    let mut rows = conn
        .query(
            r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?1 AND ts.step = ?2
               ORDER BY g.batch_index, g.group_index"#,
            libsql::params![run_id.to_string(), step],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        let gen_id: i64 = row.get(0).unwrap_or(0);
        let generation = GenerationRecord {
            batch_index: row.get::<i64>(1).unwrap_or(0),
            group_index: row.get::<i64>(2).unwrap_or(0),
            prompt: row.get::<String>(3).unwrap_or_default(),
            expected_answer: row.get::<Option<String>>(4).ok().flatten(),
            completion_text: row.get::<String>(5).unwrap_or_default(),
            completion_raw: row.get::<String>(6).unwrap_or_default(),
            thinking: row.get::<Option<String>>(7).ok().flatten(),
            num_tokens: row.get::<i64>(8).unwrap_or(0),
            finish_reason: row.get::<String>(9).unwrap_or_default(),
            reward: row.get::<f64>(10).unwrap_or(0.0),
        };

        let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Get top/bottom generations by reward
pub async fn get_generations_by_reward(
    conn: &Connection,
    run_id: &str,
    top_n: Option<i64>,
    bottom_n: Option<i64>,
    step_range: Option<Vec<i64>>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let mut results = Vec::new();

    let step_filter = if let Some(ref range) = step_range {
        if range.len() >= 2 {
            format!(" AND ts.step >= {} AND ts.step <= {}", range[0], range[1])
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Get top N
    if let Some(n) = top_n {
        let sql = format!(
            r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?1{}
               ORDER BY g.reward DESC
               LIMIT ?2"#,
            step_filter
        );

        let mut rows = conn
            .query(&sql, libsql::params![run_id.to_string(), n])
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
        {
            let gen_id: i64 = row.get(0).unwrap_or(0);
            let generation = parse_generation_row(&row)?;
            let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
            results.push(GenerationWithToolCalls {
                generation,
                tool_calls,
            });
        }
    }

    // Get bottom N
    if let Some(n) = bottom_n {
        let sql = format!(
            r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?1{}
               ORDER BY g.reward ASC
               LIMIT ?2"#,
            step_filter
        );

        let mut rows = conn
            .query(&sql, libsql::params![run_id.to_string(), n])
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
        {
            let gen_id: i64 = row.get(0).unwrap_or(0);
            let generation = parse_generation_row(&row)?;
            let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
            results.push(GenerationWithToolCalls {
                generation,
                tool_calls,
            });
        }
    }

    Ok(results)
}

/// Get generations with specific finish reason
pub async fn get_generations_by_finish_reason(
    conn: &Connection,
    run_id: &str,
    finish_reason: &str,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();

    let sql = format!(
        r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?1 AND g.finish_reason = ?2
           ORDER BY ts.step DESC{}"#,
        limit_clause
    );

    let mut rows = conn
        .query(
            &sql,
            libsql::params![run_id.to_string(), finish_reason.to_string()],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        let gen_id: i64 = row.get(0).unwrap_or(0);
        let generation = parse_generation_row(&row)?;
        let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Get generations containing tool calls
pub async fn get_generations_with_tool_calls(
    conn: &Connection,
    run_id: &str,
    tool_name: Option<String>,
    status: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();

    let mut sql = r#"SELECT DISTINCT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           JOIN tool_calls tc ON tc.generation_id = g.id
           WHERE ts.run_id = ?1"#.to_string();

    let mut param_idx = 2;
    if tool_name.is_some() {
        sql.push_str(&format!(" AND tc.tool_name = ?{}", param_idx));
        param_idx += 1;
    }
    if status.is_some() {
        sql.push_str(&format!(" AND tc.status = ?{}", param_idx));
    }

    sql.push_str(&format!(" ORDER BY ts.step DESC{}", limit_clause));

    let mut rows = match (&tool_name, &status) {
        (Some(t), Some(s)) => {
            conn.query(
                &sql,
                libsql::params![run_id.to_string(), t.clone(), s.clone()],
            )
            .await
        }
        (Some(t), None) => {
            conn.query(&sql, libsql::params![run_id.to_string(), t.clone()])
                .await
        }
        (None, Some(s)) => {
            conn.query(&sql, libsql::params![run_id.to_string(), s.clone()])
                .await
        }
        (None, None) => conn.query(&sql, libsql::params![run_id.to_string()]).await,
    }
    .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        let gen_id: i64 = row.get(0).unwrap_or(0);
        let generation = parse_generation_row(&row)?;
        let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Search generations by text content
pub async fn search_generations(
    conn: &Connection,
    run_id: &str,
    query: &str,
    search_in: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>> {
    let search_pattern = format!("%{}%", query);
    let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();

    let where_clause = match search_in.as_deref() {
        Some("prompt") => "g.prompt LIKE ?2",
        Some("completion") => "g.completion_text LIKE ?2",
        Some("thinking") => "g.thinking LIKE ?2",
        _ => "(g.prompt LIKE ?2 OR g.completion_text LIKE ?2 OR g.thinking LIKE ?2)",
    };

    let sql = format!(
        r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?1 AND {}
           ORDER BY ts.step DESC{}"#,
        where_clause, limit_clause
    );

    let mut rows = conn
        .query(
            &sql,
            libsql::params![run_id.to_string(), search_pattern.clone()],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        let gen_id: i64 = row.get(0).unwrap_or(0);
        let generation = parse_generation_row(&row)?;
        let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Get reward distribution statistics
pub async fn get_reward_stats(
    conn: &Connection,
    run_id: &str,
    step_range: Option<Vec<i64>>,
) -> Result<RewardStats> {
    let step_filter = if let Some(ref range) = step_range {
        if range.len() >= 2 {
            format!(" AND ts.step >= {} AND ts.step <= {}", range[0], range[1])
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Get basic stats
    let sql = format!(
        r#"SELECT
            COUNT(*) as count,
            AVG(g.reward) as mean,
            MIN(g.reward) as min,
            MAX(g.reward) as max
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?1{}"#,
        step_filter
    );

    let mut rows = conn
        .query(&sql, libsql::params![run_id.to_string()])
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let row = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
        .ok_or_else(|| Error::new(Status::GenericFailure, "No stats returned"))?;

    let count: i64 = row.get(0).unwrap_or(0);
    let mean: f64 = row.get(1).unwrap_or(0.0);
    let min: f64 = row.get(2).unwrap_or(0.0);
    let max: f64 = row.get(3).unwrap_or(0.0);

    if count == 0 {
        return Ok(RewardStats::default());
    }

    // Get all rewards for std dev and percentiles
    let sql = format!(
        r#"SELECT g.reward FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?1{}
           ORDER BY g.reward"#,
        step_filter
    );

    let mut rows = conn
        .query(&sql, libsql::params![run_id.to_string()])
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut rewards = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        rewards.push(row.get::<f64>(0).unwrap_or(0.0));
    }

    // Calculate std dev
    let variance: f64 = rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / count as f64;
    let std = variance.sqrt();

    // Calculate percentiles
    let n = rewards.len();
    let p25 = if n > 0 { rewards[n / 4] } else { 0.0 };
    let median = if n > 0 { rewards[n / 2] } else { 0.0 };
    let p75 = if n > 0 { rewards[3 * n / 4] } else { 0.0 };

    Ok(RewardStats {
        count,
        mean,
        std,
        min,
        max,
        median,
        p25,
        p75,
    })
}

/// Export to JSONL file
pub async fn export_jsonl(
    conn: &Connection,
    run_id: &str,
    output_path: &str,
    include_tool_calls: bool,
) -> Result<i64> {
    let file = File::create(output_path).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to create file: {}", e),
        )
    })?;
    let mut writer = BufWriter::new(file);

    let mut rows = conn
        .query(
            r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward,
                      ts.step, ts.loss
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?1
               ORDER BY ts.step, g.batch_index, g.group_index"#,
            libsql::params![run_id.to_string()],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut count = 0i64;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        let gen_id: i64 = row.get(0).unwrap_or(0);

        let mut record = json!({
            "step": row.get::<i64>(11).unwrap_or(0),
            "loss": row.get::<f64>(12).unwrap_or(0.0),
            "batch_index": row.get::<i64>(1).unwrap_or(0),
            "group_index": row.get::<i64>(2).unwrap_or(0),
            "prompt": row.get::<String>(3).unwrap_or_default(),
            "expected_answer": row.get::<Option<String>>(4).ok().flatten(),
            "completion_text": row.get::<String>(5).unwrap_or_default(),
            "completion_raw": row.get::<String>(6).unwrap_or_default(),
            "thinking": row.get::<Option<String>>(7).ok().flatten(),
            "num_tokens": row.get::<i64>(8).unwrap_or(0),
            "finish_reason": row.get::<String>(9).unwrap_or_default(),
            "reward": row.get::<f64>(10).unwrap_or(0.0),
        });

        if include_tool_calls {
            let tool_calls = get_tool_calls_for_generation(conn, gen_id).await?;
            record["tool_calls"] = json!(
                tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "status": tc.status,
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "raw_content": tc.raw_content,
                            "error_message": tc.error_message,
                        })
                    })
                    .collect::<Vec<_>>()
            );
        }

        let line = serde_json::to_string(&record).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("JSON serialize failed: {}", e),
            )
        })?;
        writeln!(writer, "{}", line)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Write failed: {}", e)))?;
        count += 1;
    }

    writer
        .flush()
        .map_err(|e| Error::new(Status::GenericFailure, format!("Flush failed: {}", e)))?;

    Ok(count)
}

/// Execute raw SQL query
pub async fn query_raw(conn: &Connection, sql: &str) -> Result<String> {
    let mut rows = conn
        .query(sql, ())
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        // Get column count from the row
        let mut row_data = serde_json::Map::new();
        // Note: libsql doesn't provide column count directly, so we try to read columns 0-20
        for i in 0..20 {
            if let Ok(val) = row.get::<String>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else if let Ok(val) = row.get::<i64>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else if let Ok(val) = row.get::<f64>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else {
                break;
            }
        }
        results.push(serde_json::Value::Object(row_data));
    }

    serde_json::to_string(&results).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("JSON serialize failed: {}", e),
        )
    })
}

// === Helper functions ===

async fn get_tool_calls_for_generation(
    conn: &Connection,
    gen_id: i64,
) -> Result<Vec<ToolCallRecord>> {
    let mut rows = conn
        .query(
            "SELECT call_index, status, tool_name, arguments, raw_content, error_message FROM tool_calls WHERE generation_id = ?1 ORDER BY call_index",
            libsql::params![gen_id],
        )
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Query failed: {}", e)))?;

    let mut tool_calls = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Row read failed: {}", e)))?
    {
        tool_calls.push(ToolCallRecord {
            call_index: row.get::<i64>(0).unwrap_or(0),
            status: row.get::<String>(1).unwrap_or_default(),
            tool_name: row.get::<Option<String>>(2).ok().flatten(),
            arguments: row.get::<Option<String>>(3).ok().flatten(),
            raw_content: row.get::<String>(4).unwrap_or_default(),
            error_message: row.get::<Option<String>>(5).ok().flatten(),
        });
    }

    Ok(tool_calls)
}

fn parse_generation_row(row: &libsql::Row) -> Result<GenerationRecord> {
    Ok(GenerationRecord {
        batch_index: row.get::<i64>(1).unwrap_or(0),
        group_index: row.get::<i64>(2).unwrap_or(0),
        prompt: row.get::<String>(3).unwrap_or_default(),
        expected_answer: row.get::<Option<String>>(4).ok().flatten(),
        completion_text: row.get::<String>(5).unwrap_or_default(),
        completion_raw: row.get::<String>(6).unwrap_or_default(),
        thinking: row.get::<Option<String>>(7).ok().flatten(),
        num_tokens: row.get::<i64>(8).unwrap_or(0),
        finish_reason: row.get::<String>(9).unwrap_or_default(),
        reward: row.get::<f64>(10).unwrap_or(0.0),
    })
}
