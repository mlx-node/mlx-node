use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const FAMILY: &str = "xlm_roberta_reranker";
pub const PROFILE: &str = "xlmr-cross-encoder-v1";
pub const PAIR_SPECIAL_TOKENS: usize = 4;
pub(super) fn invalid(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("rerank_invalid_model: {message}"))
}
fn invalid_policy(message: &str) -> Error {
    Error::from_reason(format!("rerank_invalid_request: {message}"))
}
#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub vocab_size: i64,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: i64,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}
impl Config {
    pub fn parse(raw: &Value) -> Result<Self> {
        if raw["model_type"] != "xlm-roberta"
            || !raw["architectures"]
                .as_array()
                .is_some_and(|a| a.len() == 1 && a[0] == "XLMRobertaForSequenceClassification")
        {
            return Err(invalid(
                "expected XLMRobertaForSequenceClassification architecture",
            ));
        }
        let c: Self = serde_json::from_value(raw.clone()).map_err(invalid)?;
        if [
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads,
            c.vocab_size,
            c.type_vocab_size,
        ]
        .iter()
        .any(|&v| v <= 0 || v > i32::MAX as i64)
            || c.hidden_size % c.num_attention_heads != 0
            || c.num_hidden_layers == 0
            || c.num_hidden_layers > 1024
            || c.max_position_embeddings > i32::MAX as usize
            || c.max_position_embeddings <= c.pad_token_id as usize + 1 + PAIR_SPECIAL_TOKENS
            || !c.layer_norm_eps.is_finite()
            || c.layer_norm_eps <= 0.0
            || [c.pad_token_id, c.bos_token_id, c.eos_token_id]
                .iter()
                .any(|&id| i64::from(id) >= c.vocab_size)
            || c.pad_token_id == c.bos_token_id
            || c.pad_token_id == c.eos_token_id
            || c.bos_token_id == c.eos_token_id
        {
            return Err(invalid("invalid XLM-R dimensions or special token IDs"));
        }
        if raw["hidden_act"] != "gelu"
            || raw
                .get("position_embedding_type")
                .is_some_and(|v| v != "absolute")
            || ["is_decoder", "add_cross_attention", "is_encoder_decoder"]
                .iter()
                .any(|k| raw.get(k).is_some_and(|v| v != false))
        {
            return Err(invalid(
                "profile requires an absolute-position encoder with exact GELU",
            ));
        }
        // HF config often stores id2label instead of num_labels. The tensor
        // adapter separately verifies the single-output classification head.
        if raw.get("num_labels").is_some_and(|v| v != 1)
            || raw.get("id2label").is_some_and(|v| {
                !v.as_object()
                    .is_some_and(|o| o.len() == 1 && o.contains_key("0"))
            })
        {
            return Err(invalid(
                "reranker requires exactly one classification output",
            ));
        }
        if let Some(meta) = raw.get("reranker")
            && (meta["version"] != 1
                || meta["profile"] != PROFILE
                || meta.get("family").is_some_and(|v| v != FAMILY))
        {
            return Err(invalid("conflicting or unsupported reranker metadata"));
        }
        Ok(c)
    }
    pub fn context(&self) -> usize {
        self.max_position_embeddings - self.pad_token_id as usize - 1
    }
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Options {
    pub max_pair_tokens: Option<usize>,
    pub max_query_tokens: Option<usize>,
    pub max_document_tokens: Option<usize>,
    pub max_batch_size: Option<usize>,
    pub max_batch_tokens: Option<usize>,
    pub max_documents: Option<usize>,
    pub max_input_tokens: Option<usize>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Policy {
    pub max_pair_tokens: usize,
    pub max_query_tokens: usize,
    pub max_document_tokens: usize,
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_documents: usize,
    pub max_input_tokens: usize,
}
impl Policy {
    pub fn resolve(o: Options, context: usize) -> Result<Self> {
        let p = Self {
            max_pair_tokens: o.max_pair_tokens.unwrap_or(8192.min(context)),
            max_query_tokens: o
                .max_query_tokens
                .unwrap_or(512.min(context.saturating_sub(PAIR_SPECIAL_TOKENS))),
            max_document_tokens: o.max_document_tokens.unwrap_or(8192),
            max_batch_size: o.max_batch_size.unwrap_or(8),
            max_batch_tokens: o.max_batch_tokens.unwrap_or(8192),
            max_documents: o.max_documents.unwrap_or(1024),
            max_input_tokens: o.max_input_tokens.unwrap_or(1_000_000),
        };
        if [
            p.max_pair_tokens,
            p.max_query_tokens,
            p.max_document_tokens,
            p.max_batch_size,
            p.max_batch_tokens,
            p.max_documents,
            p.max_input_tokens,
        ]
        .iter()
        .any(|&n| n == 0 || n > i32::MAX as usize)
        {
            return Err(invalid_policy(
                "policy limits must be positive 32-bit integers",
            ));
        }
        if p.max_pair_tokens > context
            || o.max_document_tokens.is_some_and(|n| n > context)
            || p.max_pair_tokens <= PAIR_SPECIAL_TOKENS
            || p.max_query_tokens > p.max_pair_tokens - PAIR_SPECIAL_TOKENS
            || p.max_batch_tokens < p.max_pair_tokens
        {
            return Err(invalid_policy(
                "pair/query/document/batch limits conflict with the model context or pair overhead",
            ));
        }
        Ok(p)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_document_limit_respects_effective_context() {
        assert_eq!(
            Policy::resolve(Options::default(), 64)
                .unwrap()
                .max_document_tokens,
            8192
        );
        assert_eq!(
            Policy::resolve(
                Options {
                    max_document_tokens: Some(64),
                    ..Default::default()
                },
                64
            )
            .unwrap()
            .max_document_tokens,
            64
        );
        assert!(
            Policy::resolve(
                Options {
                    max_document_tokens: Some(65),
                    ..Default::default()
                },
                64
            )
            .is_err()
        );
    }

    #[test]
    fn policy_rejects_conflicting_or_foreign_fields() {
        assert!(
            serde_json::from_value::<Options>(serde_json::json!({"maxBlockTokens":100})).is_err()
        );
        assert!(
            Policy::resolve(
                Options {
                    max_pair_tokens: Some(8193),
                    ..Default::default()
                },
                8192
            )
            .is_err()
        );
        assert!(
            Policy::resolve(
                Options {
                    max_pair_tokens: Some(512),
                    ..Default::default()
                },
                8192
            )
            .is_err()
        );
        assert!(
            Policy::resolve(
                Options {
                    max_batch_tokens: Some(512),
                    ..Default::default()
                },
                8192
            )
            .is_err()
        );
        assert_eq!(
            Policy::resolve(Options::default(), 8192)
                .unwrap()
                .max_pair_tokens,
            8192
        );
    }
}
