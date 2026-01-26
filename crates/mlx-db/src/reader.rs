//! Reader operations for the output store
//!
//! Handles querying training runs, steps, generations, and tool calls.

use std::fs::File;
use std::io::{BufWriter, Write};

use serde_json::json;
use sqlx::{QueryBuilder, Sqlite, SqlitePool};

use crate::error::DbError;
use crate::types::*;

/// List all training runs
pub async fn list_runs(
    pool: &SqlitePool,
    limit: Option<i64>,
    status: Option<&str>,
) -> Result<Vec<TrainingRunRecord>, DbError> {
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, name, model_name, model_path, config, started_at, ended_at, total_steps, status FROM training_runs",
    );

    if let Some(st) = status {
        builder.push(" WHERE status = ");
        builder.push_bind(st);
    }

    builder.push(" ORDER BY started_at DESC");

    if let Some(lim) = limit {
        builder.push(" LIMIT ");
        builder.push_bind(lim);
    }

    let rows: Vec<TrainingRunRow> = builder.build_query_as().fetch_all(pool).await?;

    Ok(rows.into_iter().map(|r| r.into()).collect())
}

/// Get a specific run by ID
pub async fn get_run(
    pool: &SqlitePool,
    run_id: &str,
) -> Result<Option<TrainingRunRecord>, DbError> {
    let row: Option<TrainingRunRow> = sqlx::query_as(
        "SELECT id, name, model_name, model_path, config, started_at, ended_at, total_steps, status FROM training_runs WHERE id = ?",
    )
    .bind(run_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.into()))
}

/// Find a run by name
pub async fn find_run_by_name(
    pool: &SqlitePool,
    name: &str,
) -> Result<Option<TrainingRunRecord>, DbError> {
    let row: Option<TrainingRunRow> = sqlx::query_as(
        "SELECT id, name, model_name, model_path, config, started_at, ended_at, total_steps, status FROM training_runs WHERE name = ?",
    )
    .bind(name)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.into()))
}

/// Get step summaries for a run
pub async fn get_step_summaries(
    pool: &SqlitePool,
    run_id: &str,
    start_step: Option<i64>,
    end_step: Option<i64>,
) -> Result<Vec<StepSummary>, DbError> {
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"SELECT
            ts.step,
            ts.loss,
            ts.mean_reward,
            COALESCE(ts.mean_advantage, 0.0) as mean_advantage,
            COALESCE(ts.std_advantage, 0.0) as std_advantage,
            COUNT(g.id) as num_generations,
            (SELECT COUNT(*) FROM tool_calls tc WHERE tc.generation_id IN (SELECT id FROM generations WHERE step_id = ts.id)) as num_tool_calls,
            SUM(CASE WHEN g.finish_reason = 'eos' THEN 1 ELSE 0 END) as eos_count,
            SUM(CASE WHEN g.finish_reason = 'length' THEN 1 ELSE 0 END) as length_count
        FROM training_steps ts
        LEFT JOIN generations g ON g.step_id = ts.id
        WHERE ts.run_id = "#,
    );
    builder.push_bind(run_id);

    if let Some(s) = start_step {
        builder.push(" AND ts.step >= ");
        builder.push_bind(s);
    }
    if let Some(e) = end_step {
        builder.push(" AND ts.step <= ");
        builder.push_bind(e);
    }

    builder.push(" GROUP BY ts.id ORDER BY ts.step");

    let rows: Vec<StepSummaryRow> = builder.build_query_as().fetch_all(pool).await?;

    Ok(rows.into_iter().map(|r| r.into()).collect())
}

/// Get all generations for a step
pub async fn get_generations(
    pool: &SqlitePool,
    run_id: &str,
    step: i64,
) -> Result<Vec<GenerationWithToolCalls>, DbError> {
    let rows: Vec<GenerationRow> = sqlx::query_as(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ? AND ts.step = ?
           ORDER BY g.batch_index, g.group_index"#,
    )
    .bind(run_id)
    .bind(step)
    .fetch_all(pool)
    .await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let gen_id = row.id;
        let generation: GenerationRecord = row.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Get generations with filtering
pub async fn get_generations_filtered(
    pool: &SqlitePool,
    run_id: &str,
    filter: &GenerationFilter,
) -> Result<(Vec<GenerationWithToolCalls>, usize), DbError> {
    // Build WHERE clause using QueryBuilder for count query
    let mut count_builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"SELECT COUNT(*) FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = "#,
    );
    count_builder.push_bind(run_id);
    append_generation_filters(&mut count_builder, filter);

    let total: i64 = count_builder
        .build_query_scalar()
        .fetch_one(pool)
        .await
        .map_err(|e| DbError::Query(format!("Count query failed: {}", e)))?;

    // Build main query
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = "#,
    );
    builder.push_bind(run_id);
    append_generation_filters(&mut builder, filter);

    builder.push(" ORDER BY ts.step DESC, g.reward DESC");

    if let Some(limit) = filter.limit {
        builder.push(" LIMIT ");
        builder.push_bind(limit);
    }
    if let Some(offset) = filter.offset {
        builder.push(" OFFSET ");
        builder.push_bind(offset);
    }

    let rows: Vec<GenerationRow> = builder.build_query_as().fetch_all(pool).await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let gen_id = row.id;
        let generation: GenerationRecord = row.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok((results, total as usize))
}

/// Helper function to append generation filter conditions
fn append_generation_filters(builder: &mut QueryBuilder<Sqlite>, filter: &GenerationFilter) {
    if let Some(min) = filter.step_min {
        builder.push(" AND ts.step >= ");
        builder.push_bind(min);
    }
    if let Some(max) = filter.step_max {
        builder.push(" AND ts.step <= ");
        builder.push_bind(max);
    }
    if let Some(min) = filter.reward_min {
        builder.push(" AND g.reward >= ");
        builder.push_bind(min);
    }
    if let Some(max) = filter.reward_max {
        builder.push(" AND g.reward <= ");
        builder.push_bind(max);
    }
    if let Some(ref reason) = filter.finish_reason {
        builder.push(" AND g.finish_reason = ");
        builder.push_bind(reason.clone());
    }
    if let Some(has_tools) = filter.has_tool_calls {
        if has_tools {
            builder.push(" AND EXISTS (SELECT 1 FROM tool_calls tc WHERE tc.generation_id = g.id)");
        } else {
            builder.push(
                " AND NOT EXISTS (SELECT 1 FROM tool_calls tc WHERE tc.generation_id = g.id)",
            );
        }
    }
    if let Some(ref status) = filter.tool_call_status {
        builder.push(
            " AND EXISTS (SELECT 1 FROM tool_calls tc WHERE tc.generation_id = g.id AND tc.status = ",
        );
        builder.push_bind(status.clone());
        builder.push(")");
    }
}

/// Get top/bottom generations by reward
pub async fn get_generations_by_reward(
    pool: &SqlitePool,
    run_id: &str,
    top_n: Option<i64>,
    bottom_n: Option<i64>,
    step_range: Option<&[i64]>,
) -> Result<Vec<GenerationWithToolCalls>, DbError> {
    let mut results = Vec::new();

    let step_filter = if let Some(range) = step_range {
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
            r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?{}
               ORDER BY g.reward DESC
               LIMIT ?"#,
            step_filter
        );

        let rows: Vec<GenerationRow> = sqlx::query_as(&sql)
            .bind(run_id)
            .bind(n)
            .fetch_all(pool)
            .await?;

        for row in rows {
            let gen_id = row.id;
            let generation: GenerationRecord = row.into();
            let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
            results.push(GenerationWithToolCalls {
                generation,
                tool_calls,
            });
        }
    }

    // Get bottom N
    if let Some(n) = bottom_n {
        let sql = format!(
            r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                      g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
               FROM generations g
               JOIN training_steps ts ON g.step_id = ts.id
               WHERE ts.run_id = ?{}
               ORDER BY g.reward ASC
               LIMIT ?"#,
            step_filter
        );

        let rows: Vec<GenerationRow> = sqlx::query_as(&sql)
            .bind(run_id)
            .bind(n)
            .fetch_all(pool)
            .await?;

        for row in rows {
            let gen_id = row.id;
            let generation: GenerationRecord = row.into();
            let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
            results.push(GenerationWithToolCalls {
                generation,
                tool_calls,
            });
        }
    }

    Ok(results)
}

/// Get a single generation with its tool calls by ID
pub async fn get_generation_by_id(
    pool: &SqlitePool,
    gen_id: i64,
) -> Result<Option<GenerationWithToolCalls>, DbError> {
    let row: Option<GenerationRow> = sqlx::query_as(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE g.id = ?"#,
    )
    .bind(gen_id)
    .fetch_optional(pool)
    .await?;

    if let Some(r) = row {
        let generation: GenerationRecord = r.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        Ok(Some(GenerationWithToolCalls {
            generation,
            tool_calls,
        }))
    } else {
        Ok(None)
    }
}

/// Get generations with specific finish reason
pub async fn get_generations_by_finish_reason(
    pool: &SqlitePool,
    run_id: &str,
    finish_reason: &str,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>, DbError> {
    let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();

    let sql = format!(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ? AND g.finish_reason = ?
           ORDER BY ts.step DESC{}"#,
        limit_clause
    );

    let rows: Vec<GenerationRow> = sqlx::query_as(&sql)
        .bind(run_id)
        .bind(finish_reason)
        .fetch_all(pool)
        .await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let gen_id = row.id;
        let generation: GenerationRecord = row.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Get generations containing tool calls
pub async fn get_generations_with_tool_calls(
    pool: &SqlitePool,
    run_id: &str,
    tool_name: Option<&str>,
    status: Option<&str>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>, DbError> {
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"SELECT DISTINCT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           JOIN tool_calls tc ON tc.generation_id = g.id
           WHERE ts.run_id = "#,
    );
    builder.push_bind(run_id);

    if let Some(t) = tool_name {
        builder.push(" AND tc.tool_name = ");
        builder.push_bind(t);
    }
    if let Some(s) = status {
        builder.push(" AND tc.status = ");
        builder.push_bind(s);
    }

    builder.push(" ORDER BY ts.step DESC");

    if let Some(lim) = limit {
        builder.push(" LIMIT ");
        builder.push_bind(lim);
    }

    let rows: Vec<GenerationRow> = builder.build_query_as().fetch_all(pool).await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let gen_id = row.id;
        let generation: GenerationRecord = row.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Search generations by text content
pub async fn search_generations(
    pool: &SqlitePool,
    run_id: &str,
    query: &str,
    search_in: Option<&str>,
    limit: Option<i64>,
) -> Result<Vec<GenerationWithToolCalls>, DbError> {
    let search_pattern = format!("%{}%", query);
    let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();

    let where_clause = match search_in {
        Some("prompt") => "g.prompt LIKE ?",
        Some("completion") => "g.completion_text LIKE ?",
        Some("thinking") => "g.thinking LIKE ?",
        _ => "(g.prompt LIKE ? OR g.completion_text LIKE ? OR g.thinking LIKE ?)",
    };

    let sql = format!(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ? AND {}
           ORDER BY ts.step DESC{}"#,
        where_clause, limit_clause
    );

    let rows: Vec<GenerationRow> = match search_in {
        Some("prompt") | Some("completion") | Some("thinking") => {
            sqlx::query_as(&sql)
                .bind(run_id)
                .bind(&search_pattern)
                .fetch_all(pool)
                .await?
        }
        _ => {
            sqlx::query_as(&sql)
                .bind(run_id)
                .bind(&search_pattern)
                .bind(&search_pattern)
                .bind(&search_pattern)
                .fetch_all(pool)
                .await?
        }
    };

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let gen_id = row.id;
        let generation: GenerationRecord = row.into();
        let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
        results.push(GenerationWithToolCalls {
            generation,
            tool_calls,
        });
    }

    Ok(results)
}

/// Helper struct for reward stats query
#[derive(sqlx::FromRow)]
struct RewardStatsRow {
    count: i64,
    mean: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

/// Get reward distribution statistics
pub async fn get_reward_stats(
    pool: &SqlitePool,
    run_id: &str,
    step_range: Option<&[i64]>,
) -> Result<RewardStats, DbError> {
    let step_filter = if let Some(range) = step_range {
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
           WHERE ts.run_id = ?{}"#,
        step_filter
    );

    let row: RewardStatsRow = sqlx::query_as(&sql).bind(run_id).fetch_one(pool).await?;

    let count = row.count;
    let mean = row.mean.unwrap_or(0.0);
    let min = row.min.unwrap_or(0.0);
    let max = row.max.unwrap_or(0.0);

    if count == 0 {
        return Ok(RewardStats::default());
    }

    // Get all rewards for std dev and percentiles
    let sql = format!(
        r#"SELECT g.reward FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?{}
           ORDER BY g.reward"#,
        step_filter
    );

    let rewards: Vec<f64> = sqlx::query_scalar(&sql)
        .bind(run_id)
        .fetch_all(pool)
        .await?;

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

/// Helper struct for export row
#[derive(sqlx::FromRow)]
struct ExportRow {
    id: i64,
    batch_index: i64,
    group_index: i64,
    prompt: String,
    expected_answer: Option<String>,
    completion_text: String,
    completion_raw: String,
    thinking: Option<String>,
    num_tokens: i64,
    finish_reason: String,
    reward: f64,
    step: i64,
    loss: f64,
}

/// Export to JSONL file
pub async fn export_jsonl(
    pool: &SqlitePool,
    run_id: &str,
    output_path: &str,
    include_tool_calls: bool,
) -> Result<i64, DbError> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    let rows: Vec<ExportRow> = sqlx::query_as(
        r#"SELECT g.id, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward,
                  ts.step, ts.loss
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?
           ORDER BY ts.step, g.batch_index, g.group_index"#,
    )
    .bind(run_id)
    .fetch_all(pool)
    .await?;

    let mut count = 0i64;
    for row in rows {
        let gen_id = row.id;

        let mut record = json!({
            "step": row.step,
            "loss": row.loss,
            "batch_index": row.batch_index,
            "group_index": row.group_index,
            "prompt": row.prompt,
            "expected_answer": row.expected_answer,
            "completion_text": row.completion_text,
            "completion_raw": row.completion_raw,
            "thinking": row.thinking,
            "num_tokens": row.num_tokens,
            "finish_reason": row.finish_reason,
            "reward": row.reward,
        });

        if include_tool_calls {
            let tool_calls = get_tool_calls_for_generation(pool, gen_id).await?;
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

        let line = serde_json::to_string(&record)?;
        writeln!(writer, "{}", line)?;
        count += 1;
    }

    writer.flush()?;

    Ok(count)
}

/// Execute raw SQL query
pub async fn query_raw(pool: &SqlitePool, sql: &str) -> Result<String, DbError> {
    // For raw queries, we need to handle dynamic columns
    // sqlx doesn't support truly dynamic queries, so we use a workaround
    // by fetching rows as raw sqlite rows
    use sqlx::Row;

    let rows = sqlx::query(sql).fetch_all(pool).await?;

    let mut results = Vec::new();
    for row in rows {
        let mut row_data = serde_json::Map::new();
        // Try to read columns 0-19
        for i in 0..20 {
            if let Ok(val) = row.try_get::<String, _>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else if let Ok(val) = row.try_get::<i64, _>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else if let Ok(val) = row.try_get::<f64, _>(i) {
                row_data.insert(format!("col{}", i), json!(val));
            } else {
                break;
            }
        }
        results.push(serde_json::Value::Object(row_data));
    }

    Ok(serde_json::to_string(&results)?)
}

// === Log functions ===

/// Get logs with optional filtering
pub async fn get_logs(
    pool: &SqlitePool,
    run_id: Option<&str>,
    min_level: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<LogRecord>, DbError> {
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, run_id, level, target, message, file, line, created_at FROM logs WHERE 1=1",
    );

    if let Some(rid) = run_id {
        builder.push(" AND run_id = ");
        builder.push_bind(rid);
    }

    if let Some(level) = min_level {
        // Filter by level priority: ERROR > WARN > INFO > DEBUG
        let levels = match level.to_uppercase().as_str() {
            "ERROR" => vec!["ERROR"],
            "WARN" => vec!["ERROR", "WARN"],
            "INFO" => vec!["ERROR", "WARN", "INFO"],
            _ => vec!["ERROR", "WARN", "INFO", "DEBUG"],
        };
        builder.push(" AND level IN (");
        let mut separated = builder.separated(", ");
        for l in levels {
            separated.push_bind(l);
        }
        separated.push_unseparated(")");
    }

    builder.push(" ORDER BY created_at DESC LIMIT ");
    builder.push_bind(limit);
    builder.push(" OFFSET ");
    builder.push_bind(offset);

    let rows: Vec<LogRow> = builder.build_query_as().fetch_all(pool).await?;

    Ok(rows.into_iter().map(|r| r.into()).collect())
}

/// Get count of logs matching filter
pub async fn get_log_count(
    pool: &SqlitePool,
    run_id: Option<&str>,
    min_level: Option<&str>,
) -> Result<i64, DbError> {
    let mut builder: QueryBuilder<Sqlite> =
        QueryBuilder::new("SELECT COUNT(*) FROM logs WHERE 1=1");

    if let Some(rid) = run_id {
        builder.push(" AND run_id = ");
        builder.push_bind(rid);
    }

    if let Some(level) = min_level {
        let levels = match level.to_uppercase().as_str() {
            "ERROR" => vec!["ERROR"],
            "WARN" => vec!["ERROR", "WARN"],
            "INFO" => vec!["ERROR", "WARN", "INFO"],
            _ => vec!["ERROR", "WARN", "INFO", "DEBUG"],
        };
        builder.push(" AND level IN (");
        let mut separated = builder.separated(", ");
        for l in levels {
            separated.push_bind(l);
        }
        separated.push_unseparated(")");
    }

    let count: i64 = builder.build_query_scalar().fetch_one(pool).await?;

    Ok(count)
}

// === Helper functions ===

/// Get tool calls for a specific generation
pub async fn get_tool_calls_for_generation(
    pool: &SqlitePool,
    gen_id: i64,
) -> Result<Vec<ToolCallRecord>, DbError> {
    let rows: Vec<ToolCallRow> = sqlx::query_as(
        "SELECT id, call_index, status, tool_name, arguments, raw_content, error_message FROM tool_calls WHERE generation_id = ? ORDER BY call_index",
    )
    .bind(gen_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|r| r.into_record(gen_id)).collect())
}

// =============================================================================
// TUI Resume State Functions
// =============================================================================

/// Get recent step metrics for sparkline restoration
///
/// Returns metrics ordered by step (oldest first) for easy insertion into VecDeque.
/// Useful for restoring TUI sparklines on resume.
pub async fn get_recent_step_metrics(
    pool: &SqlitePool,
    run_id: &str,
    limit: i64,
) -> Result<Vec<StepMetricSummary>, DbError> {
    // Query with subquery to get last N steps, then order ascending
    let rows: Vec<StepMetricRow> = sqlx::query_as(
        r#"SELECT step, loss, mean_reward, COALESCE(mean_advantage, 0.0) as mean_advantage,
                  COALESCE(std_advantage, 0.0) as std_advantage,
                  total_tokens, generation_time_ms, training_time_ms
           FROM training_steps
           WHERE run_id = ?
           ORDER BY step DESC
           LIMIT ?"#,
    )
    .bind(run_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    // Reverse to get oldest-first order (for VecDeque push_back)
    let mut summaries: Vec<StepMetricSummary> = rows.into_iter().map(|r| r.into()).collect();
    summaries.reverse();

    Ok(summaries)
}

/// Get aggregate statistics for a training run
///
/// Returns pre-computed aggregates that can be used to restore TUI state on resume.
pub async fn get_run_aggregates(pool: &SqlitePool, run_id: &str) -> Result<RunAggregates, DbError> {
    let row: Option<RunAggregatesRow> = sqlx::query_as(
        r#"SELECT
            MAX(mean_reward) as best_reward,
            AVG(mean_reward) as avg_reward,
            COUNT(*) as reward_count,
            MIN(loss) as best_loss,
            AVG(loss) as avg_loss,
            COUNT(*) as loss_count,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            COALESCE(MAX(step), 0) as current_step,
            COALESCE(AVG(generation_time_ms), 0.0) as avg_generation_time_ms,
            COALESCE(AVG(training_time_ms), 0.0) as avg_training_time_ms
           FROM training_steps
           WHERE run_id = ?"#,
    )
    .bind(run_id)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(r.into()),
        None => Ok(RunAggregates::default()),
    }
}

/// Get recent generations for sample panel restoration
///
/// Returns generations ordered by step DESC, reward DESC (most recent high-reward first).
pub async fn get_recent_generations(
    pool: &SqlitePool,
    run_id: &str,
    limit: i64,
) -> Result<Vec<GenerationRecord>, DbError> {
    let rows: Vec<GenerationRow> = sqlx::query_as(
        r#"SELECT g.id, ts.step, g.batch_index, g.group_index, g.prompt, g.expected_answer,
                  g.completion_text, g.completion_raw, g.thinking, g.num_tokens, g.finish_reason, g.reward
           FROM generations g
           JOIN training_steps ts ON g.step_id = ts.id
           WHERE ts.run_id = ?
           ORDER BY ts.step DESC, g.reward DESC
           LIMIT ?"#,
    )
    .bind(run_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|r| r.into()).collect())
}
