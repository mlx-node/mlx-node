use napi::bindgen_prelude::*;
use napi_derive::napi;
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;

use mlx_db::response_store::types::StoredResponse;
use mlx_db::response_store::{reader, schema, writer};

use super::types::StoredResponseRecord;

/// Response store for OpenAI Responses API persistence.
///
/// Stores responses in SQLite to support `previous_response_id`
/// for multi-turn conversation state.
#[napi]
pub struct ResponseStore {
    pool: SqlitePool,
}

#[napi]
impl ResponseStore {
    /// Open (or create) a response store at the given path.
    #[napi(factory)]
    pub async fn open(path: String) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", path);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to open response store: {}", e),
                )
            })?;

        schema::init_response_schema(&pool).await.map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to init response schema: {}", e),
            )
        })?;

        Ok(Self { pool })
    }

    /// Store a response.
    #[napi]
    pub async fn store(&self, response: StoredResponseRecord) -> Result<()> {
        let stored = StoredResponse {
            id: response.id,
            created_at: response.created_at,
            model: response.model,
            status: response.status,
            instructions: response.instructions,
            input_json: response.input_json,
            output_json: response.output_json,
            output_text: response.output_text,
            usage_json: response.usage_json,
            previous_response_id: response.previous_response_id,
            config_json: response.config_json,
            expires_at: response.expires_at,
        };
        writer::store_response(&self.pool, &stored)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(())
    }

    /// Get a single response by ID.
    #[napi]
    pub async fn get(&self, id: String) -> Result<Option<StoredResponseRecord>> {
        let response = reader::get_response(&self.pool, &id)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(response.map(|r| r.into()))
    }

    /// Get the full conversation chain for a response (oldest first).
    #[napi]
    pub async fn get_chain(&self, id: String) -> Result<Vec<StoredResponseRecord>> {
        let chain = reader::get_response_chain(&self.pool, &id)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(chain.into_iter().map(|r| r.into()).collect())
    }

    /// Delete a response by ID. Returns true if a row was deleted.
    #[napi]
    pub async fn delete(&self, id: String) -> Result<bool> {
        writer::delete_response(&self.pool, &id)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }

    /// Delete expired responses. Returns the number of rows deleted.
    #[napi]
    pub async fn cleanup_expired(&self) -> Result<u32> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        writer::cleanup_expired(&self.pool, now)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }
}
