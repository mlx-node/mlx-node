use sqlx::SqlitePool;

use crate::error::DbError;

pub const CREATE_RESPONSE_TABLES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    instructions TEXT,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    output_text TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    previous_response_id TEXT,
    config_json TEXT,
    expires_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);
CREATE INDEX IF NOT EXISTS idx_responses_expires ON responses(expires_at);
"#;

pub async fn init_response_schema(pool: &SqlitePool) -> Result<(), DbError> {
    for statement in CREATE_RESPONSE_TABLES_SQL.split(';') {
        let trimmed = statement.trim();
        if trimmed.is_empty() {
            continue;
        }
        sqlx::query(trimmed)
            .execute(pool)
            .await
            .map_err(|e| DbError::Schema(format!("Failed to create response schema: {}", e)))?;
    }
    Ok(())
}
