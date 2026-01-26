//! Database schema for the output store
//!
//! Defines tables for storing training runs, steps, generations, and tool calls.

use crate::error::DbError;
use sqlx::SqlitePool;

/// SQL statements for creating the schema
pub const CREATE_TABLES_SQL: &str = r#"
-- Training runs (one per training session)
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    name TEXT,
    model_name TEXT NOT NULL,
    model_path TEXT,
    config TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    total_steps INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

-- Individual training steps
CREATE TABLE IF NOT EXISTS training_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id),
    step INTEGER NOT NULL,
    epoch INTEGER,
    loss REAL NOT NULL,
    mean_reward REAL NOT NULL,
    std_reward REAL NOT NULL,
    mean_advantage REAL,
    std_advantage REAL DEFAULT 0.0,
    total_tokens INTEGER,
    generation_time_ms REAL,
    training_time_ms REAL,
    gradients_applied INTEGER,
    created_at INTEGER NOT NULL,
    UNIQUE(run_id, step)
);

-- All generations (group_size * batch_size per step)
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id),
    step_id INTEGER NOT NULL REFERENCES training_steps(id),
    batch_index INTEGER NOT NULL,
    group_index INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    expected_answer TEXT,
    completion_text TEXT NOT NULL,
    completion_raw TEXT NOT NULL,
    thinking TEXT,
    num_tokens INTEGER NOT NULL,
    finish_reason TEXT NOT NULL,
    reward REAL NOT NULL,
    created_at INTEGER NOT NULL
);

-- Parsed tool calls (0-N per generation)
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),
    call_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    tool_name TEXT,
    arguments TEXT,
    raw_content TEXT NOT NULL,
    error_message TEXT
);

-- Application logs (persisted from TUI)
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES training_runs(id),
    level TEXT NOT NULL,
    target TEXT NOT NULL,
    message TEXT NOT NULL,
    file TEXT,
    line INTEGER,
    created_at INTEGER NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_steps_run ON training_steps(run_id, step);
CREATE INDEX IF NOT EXISTS idx_gens_step ON generations(step_id);
CREATE INDEX IF NOT EXISTS idx_gens_run_reward ON generations(run_id, reward);
CREATE INDEX IF NOT EXISTS idx_tool_calls_gen ON tool_calls(generation_id);
CREATE INDEX IF NOT EXISTS idx_gens_finish ON generations(run_id, finish_reason);
CREATE INDEX IF NOT EXISTS idx_runs_name ON training_runs(name);
CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
"#;

/// Initialize the database schema
pub async fn init_schema(pool: &SqlitePool) -> Result<(), DbError> {
    // Split SQL into individual statements and execute each
    for statement in CREATE_TABLES_SQL.split(';') {
        // Remove comment lines and trim
        let cleaned: String = statement
            .lines()
            .filter(|line| !line.trim().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        let cleaned = cleaned.trim();

        if !cleaned.is_empty() {
            sqlx::query(cleaned)
                .execute(pool)
                .await
                .map_err(|e| DbError::Schema(format!("Failed to execute statement: {}", e)))?;
        }
    }

    // Run migrations for existing databases
    migrate_schema(pool).await?;

    Ok(())
}

/// Migrate schema for existing databases (add new columns, etc.)
async fn migrate_schema(pool: &SqlitePool) -> Result<(), DbError> {
    // Check if name column exists in training_runs
    let has_name = check_column_exists(pool, "training_runs", "name").await?;
    if !has_name {
        sqlx::query("ALTER TABLE training_runs ADD COLUMN name TEXT")
            .execute(pool)
            .await
            .map_err(|e| DbError::Migration(format!("Failed to add name column: {}", e)))?;
    }

    // Check if std_advantage column exists in training_steps
    let has_std_advantage = check_column_exists(pool, "training_steps", "std_advantage").await?;
    if !has_std_advantage {
        sqlx::query("ALTER TABLE training_steps ADD COLUMN std_advantage REAL DEFAULT 0.0")
            .execute(pool)
            .await
            .map_err(|e| {
                DbError::Migration(format!("Failed to add std_advantage column: {}", e))
            })?;
    }

    Ok(())
}

/// Helper struct for PRAGMA table_info results
#[derive(sqlx::FromRow)]
struct TableInfo {
    name: String,
}

/// Check if a column exists in a table
async fn check_column_exists(
    pool: &SqlitePool,
    table: &str,
    column: &str,
) -> Result<bool, DbError> {
    let sql = format!("PRAGMA table_info({})", table);
    let rows: Vec<TableInfo> = sqlx::query_as(&sql)
        .fetch_all(pool)
        .await
        .map_err(|e| DbError::Query(format!("PRAGMA failed: {}", e)))?;

    Ok(rows.iter().any(|row| row.name == column))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tables_sql_is_valid() {
        // Basic validation that the SQL string is well-formed
        assert!(CREATE_TABLES_SQL.contains("CREATE TABLE IF NOT EXISTS training_runs"));
        assert!(CREATE_TABLES_SQL.contains("CREATE TABLE IF NOT EXISTS training_steps"));
        assert!(CREATE_TABLES_SQL.contains("CREATE TABLE IF NOT EXISTS generations"));
        assert!(CREATE_TABLES_SQL.contains("CREATE TABLE IF NOT EXISTS tool_calls"));
        assert!(CREATE_TABLES_SQL.contains("CREATE TABLE IF NOT EXISTS logs"));
        assert!(CREATE_TABLES_SQL.contains("CREATE INDEX IF NOT EXISTS"));
        // Check for new name column and index
        assert!(CREATE_TABLES_SQL.contains("name TEXT"));
        assert!(CREATE_TABLES_SQL.contains("idx_runs_name"));
        // Check for logs indexes
        assert!(CREATE_TABLES_SQL.contains("idx_logs_run"));
        assert!(CREATE_TABLES_SQL.contains("idx_logs_level"));
    }
}
