use serde::{Deserialize, Serialize};

/// A stored response record in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub instructions: Option<String>,
    pub input_json: String,
    pub output_json: String,
    pub output_text: String,
    pub usage_json: String,
    pub previous_response_id: Option<String>,
    pub config_json: Option<String>,
    pub expires_at: Option<i64>,
}

#[derive(sqlx::FromRow)]
pub(crate) struct StoredResponseRow {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub instructions: Option<String>,
    pub input_json: String,
    pub output_json: String,
    pub output_text: String,
    pub usage_json: String,
    pub previous_response_id: Option<String>,
    pub config_json: Option<String>,
    pub expires_at: Option<i64>,
}

impl From<StoredResponseRow> for StoredResponse {
    fn from(row: StoredResponseRow) -> Self {
        StoredResponse {
            id: row.id,
            created_at: row.created_at,
            model: row.model,
            status: row.status,
            instructions: row.instructions,
            input_json: row.input_json,
            output_json: row.output_json,
            output_text: row.output_text,
            usage_json: row.usage_json,
            previous_response_id: row.previous_response_id,
            config_json: row.config_json,
            expires_at: row.expires_at,
        }
    }
}
