use sqlx::SqlitePool;

use super::types::StoredResponse;
use crate::error::DbError;

pub async fn store_response(pool: &SqlitePool, response: &StoredResponse) -> Result<(), DbError> {
    sqlx::query(
        r#"INSERT INTO responses (id, created_at, model, status, instructions, input_json, output_json, output_text, usage_json, previous_response_id, config_json, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
    )
    .bind(&response.id)
    .bind(response.created_at)
    .bind(&response.model)
    .bind(&response.status)
    .bind(&response.instructions)
    .bind(&response.input_json)
    .bind(&response.output_json)
    .bind(&response.output_text)
    .bind(&response.usage_json)
    .bind(&response.previous_response_id)
    .bind(&response.config_json)
    .bind(response.expires_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_response(pool: &SqlitePool, id: &str) -> Result<bool, DbError> {
    let result = sqlx::query("DELETE FROM responses WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn cleanup_expired(pool: &SqlitePool, now: i64) -> Result<u32, DbError> {
    let result =
        sqlx::query("DELETE FROM responses WHERE expires_at IS NOT NULL AND expires_at < ?")
            .bind(now)
            .execute(pool)
            .await?;
    Ok(result.rows_affected() as u32)
}
