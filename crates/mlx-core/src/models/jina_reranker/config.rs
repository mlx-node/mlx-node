use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(super) fn invalid(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("rerank_invalid_model: {message}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Profile {
    #[serde(rename = "jina-v3")]
    V3,
    #[serde(rename = "jina-v3.5")]
    V35,
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub vocab_size: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl Config {
    pub fn parse(raw: &Value) -> Result<(Self, Profile)> {
        if !raw["architectures"]
            .as_array()
            .is_some_and(|a| a.iter().any(|v| v == "JinaForRanking"))
        {
            return Err(invalid("expected JinaForRanking architecture"));
        }
        let mut c: Self = serde_json::from_value(raw.clone()).map_err(invalid)?;
        if [
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.vocab_size,
        ]
        .iter()
        .any(|&v| v <= 0 || v > i32::MAX as i64)
            || c.num_hidden_layers == 0
            || c.num_hidden_layers > 1024
            || c.max_position_embeddings == 0
            || c.max_position_embeddings > i32::MAX as usize
            || c.num_attention_heads % c.num_key_value_heads != 0
            || c.head_dim % 2 != 0
            || !c.rms_norm_eps.is_finite()
            || c.rms_norm_eps <= 0.0
            || !c.rope_theta.is_finite()
            || c.rope_theta <= 0.0
        {
            return Err(invalid(
                "invalid dimensions, normalization or RoPE configuration",
            ));
        }
        for key in ["attention_bias", "mlp_bias"] {
            if raw.get(key).is_some_and(|v| v != false) {
                return Err(invalid(format!("unsupported {key}")));
            }
        }
        if raw.get("rope_scaling").is_some_and(|v| !v.is_null()) {
            return Err(invalid(
                "scaled RoPE is not supported by this reranker profile",
            ));
        }
        if raw.get("hidden_act").is_some_and(|v| v != "silu") {
            return Err(invalid("expected silu activation"));
        }
        let profile = if c.use_sliding_window {
            if c.sliding_window.is_none_or(|v| v <= 0)
                || c.layer_types.len() != c.num_hidden_layers
                || !c.layer_types.iter().any(|v| v == "sliding_attention")
                || !c.layer_types.iter().any(|v| v == "full_attention")
            {
                return Err(invalid(
                    "hybrid attention requires explicit full/sliding layer types and a positive window",
                ));
            }
            Profile::V35
        } else {
            if c.layer_types.is_empty() {
                c.layer_types = vec!["full_attention".into(); c.num_hidden_layers];
            }
            if c.layer_types.iter().any(|v| v != "full_attention") {
                return Err(invalid("v3 requires full attention"));
            }
            Profile::V3
        };
        if c.layer_types.len() != c.num_hidden_layers
            || c.layer_types
                .iter()
                .any(|v| !matches!(v.as_str(), "full_attention" | "sliding_attention"))
        {
            return Err(invalid("unknown or incomplete attention layout"));
        }
        if let Some(meta) = raw.get("reranker") {
            if meta["version"] != 1 {
                return Err(invalid("unsupported reranker metadata version"));
            }
            let declared: Profile =
                serde_json::from_value(meta["profile"].clone()).map_err(invalid)?;
            if declared != profile {
                return Err(invalid(
                    "reranker profile conflicts with attention configuration",
                ));
            }
        }
        Ok((c, profile))
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Options {
    pub max_block_tokens: Option<usize>,
    pub max_documents_per_block: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub max_query_tokens: Option<usize>,
    pub max_document_tokens: Option<usize>,
    pub max_documents: Option<usize>,
    pub max_input_tokens: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Policy {
    pub max_block_tokens: usize,
    pub max_documents_per_block: usize,
    pub prefill_chunk_size: usize,
    pub max_query_tokens: usize,
    pub max_document_tokens: usize,
    pub max_documents: usize,
    pub max_input_tokens: usize,
}
impl Policy {
    pub fn resolve(options: Options, profile: Profile, context: usize) -> Result<Self> {
        let p = Self {
            max_block_tokens: options.max_block_tokens.unwrap_or(32768).min(context),
            max_documents_per_block: options.max_documents_per_block.unwrap_or(125),
            prefill_chunk_size: options.prefill_chunk_size.unwrap_or(512),
            max_query_tokens: options
                .max_query_tokens
                .unwrap_or(if profile == Profile::V3 { 512 } else { 2048 }),
            max_document_tokens: options
                .max_document_tokens
                .unwrap_or(if profile == Profile::V3 { 2048 } else { 8192 }),
            max_documents: options.max_documents.unwrap_or(1024),
            max_input_tokens: options.max_input_tokens.unwrap_or(1_000_000),
        };
        if [
            p.max_block_tokens,
            p.max_documents_per_block,
            p.prefill_chunk_size,
            p.max_query_tokens,
            p.max_document_tokens,
            p.max_documents,
            p.max_input_tokens,
        ]
        .contains(&0)
        {
            return Err(Error::from_reason(
                "rerank_invalid_request: policy limits must be positive",
            ));
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw() -> Value {
        serde_json::json!({"architectures":["JinaForRanking"],"hidden_size":24,"intermediate_size":48,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":100,"rms_norm_eps":0.000001,"rope_theta":10000,"max_position_embeddings":128})
    }
    #[test]
    fn detects_geometry_without_model_names() {
        let (c, p) = Config::parse(&raw()).unwrap();
        assert_eq!(c.head_dim, 8);
        assert_eq!(p, Profile::V3);
        let mut r = raw();
        r["use_sliding_window"] = true.into();
        r["sliding_window"] = 16.into();
        r["layer_types"] = serde_json::json!(["sliding_attention", "full_attention"]);
        assert_eq!(Config::parse(&r).unwrap().1, Profile::V35);
        r["reranker"] = serde_json::json!({"version":1,"profile":"jina-v3"});
        assert!(Config::parse(&r).is_err());
    }
    #[test]
    fn refuses_unknown_or_incomplete_layout() {
        let mut r = raw();
        r["layer_types"] = serde_json::json!(["full_attention"]);
        assert!(Config::parse(&r).is_err());
        r = raw();
        r["use_sliding_window"] = true.into();
        assert!(Config::parse(&r).is_err());
    }
}
