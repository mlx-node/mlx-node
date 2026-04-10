use sqlx::SqlitePool;

use super::types::{StoredResponse, StoredResponseRow};
use crate::error::DbError;

pub async fn get_response(pool: &SqlitePool, id: &str) -> Result<Option<StoredResponse>, DbError> {
    let row: Option<StoredResponseRow> = sqlx::query_as(
        "SELECT id, created_at, model, status, instructions, input_json, output_json, output_text, usage_json, previous_response_id, config_json, expires_at FROM responses WHERE id = ? AND (expires_at IS NULL OR expires_at > unixepoch())",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| r.into()))
}

/// Follow the `previous_response_id` chain to build the full conversation history.
/// Returns responses ordered from oldest to newest.
pub async fn get_response_chain(
    pool: &SqlitePool,
    id: &str,
) -> Result<Vec<StoredResponse>, DbError> {
    let mut chain = Vec::new();
    let mut current_id = Some(id.to_string());

    while let Some(ref cid) = current_id {
        let row: Option<StoredResponseRow> = sqlx::query_as(
            "SELECT id, created_at, model, status, instructions, input_json, output_json, output_text, usage_json, previous_response_id, config_json, expires_at FROM responses WHERE id = ? AND (expires_at IS NULL OR expires_at > unixepoch())",
        )
        .bind(cid)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => {
                let response: StoredResponse = r.into();
                current_id = response.previous_response_id.clone();
                chain.push(response);
            }
            None => {
                return Err(DbError::Query(format!("Response not found: {}", cid)));
            }
        }
    }

    chain.reverse(); // oldest first
    Ok(chain)
}
