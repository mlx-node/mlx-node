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

/// Maximum chain depth to prevent infinite loops from cyclic data.
const MAX_CHAIN_DEPTH: usize = 100;

/// Follow the `previous_response_id` chain to build the full conversation history.
/// Returns responses ordered from oldest to newest.
/// Stops after `MAX_CHAIN_DEPTH` links to guard against cycles or excessively long chains.
pub async fn get_response_chain(
    pool: &SqlitePool,
    id: &str,
) -> Result<Vec<StoredResponse>, DbError> {
    let mut chain = Vec::new();
    let mut current_id = Some(id.to_string());
    let mut seen = std::collections::HashSet::new();

    while let Some(ref cid) = current_id {
        if chain.len() >= MAX_CHAIN_DEPTH {
            return Err(DbError::Query(format!(
                "Response chain exceeds maximum depth of {}",
                MAX_CHAIN_DEPTH
            )));
        }
        if !seen.insert(cid.clone()) {
            return Err(DbError::Query(format!(
                "Cycle detected in response chain at id: {}",
                cid
            )));
        }

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
