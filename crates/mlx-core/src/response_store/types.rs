use napi_derive::napi;

/// A stored response record exposed to JavaScript.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct StoredResponseRecord {
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

impl From<mlx_db::response_store::types::StoredResponse> for StoredResponseRecord {
    fn from(r: mlx_db::response_store::types::StoredResponse) -> Self {
        StoredResponseRecord {
            id: r.id,
            created_at: r.created_at,
            model: r.model,
            status: r.status,
            instructions: r.instructions,
            input_json: r.input_json,
            output_json: r.output_json,
            output_text: r.output_text,
            usage_json: r.usage_json,
            previous_response_id: r.previous_response_id,
            config_json: r.config_json,
            expires_at: r.expires_at,
        }
    }
}
