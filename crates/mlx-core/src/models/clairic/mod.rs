//! Clairic structured-extraction runtime MLX/N-API bindings.
//!
//! Pure-Rust extraction logic, grammar constraints, prompt encoding, typed
//! decoding, and orchestration live in the `mlx-clairic` workspace crate and
//! are re-exported here for internal callers.

pub mod ar_engine;
pub mod head;
#[cfg(test)]
mod orchestrator_tests;

pub use mlx_clairic::{
    ar, completion, contracts, crops, decimal_util, embedded, errors, grounding, head_config,
    html_rows, identifier_spelling, image_rows, json_validate, lists, media, media_cache,
    media_state_prompt, orchestrator, orientation, printed_numbers, printed_temporal, profile,
    receipt, row_omission, schema, scopes, settings, source_dates, source_rows, table_rows,
    typed_decode, typed_prompt, types,
};

use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use napi::Result;
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use crate::engine::checkpoint_source::CheckpointSource;
use crate::models::qwen3_5::{model::Qwen3_5Model, persistence};
use crate::utils::gguf::NativeGgufFamily;

use mlx_clairic::embedded::RenderedPdfPageData;
use mlx_clairic::profile::ClairicProfile;

/// Convert a pure-Rust [`mlx_clairic::ClairicError`] into a [`napi::Error`].
pub(crate) fn to_napi_err(err: mlx_clairic::ClairicError) -> napi::Error {
    napi::Error::from_reason(err.to_string())
}

/// Convert a [`napi::Error`] from MLX/tokenizer operations into a [`mlx_clairic::ClairicError`]
/// without double-prefixing existing Clairic error prefixes.
pub(crate) fn napi_to_clairic_err(err: napi::Error) -> mlx_clairic::ClairicError {
    let msg = err.reason;
    if msg == mlx_clairic::errors::CANCELLED_MESSAGE {
        mlx_clairic::ClairicError::Cancelled
    } else if let Some(rest) = msg.strip_prefix(mlx_clairic::errors::INVALID_REQUEST_PREFIX) {
        mlx_clairic::ClairicError::InvalidRequest(rest.to_string())
    } else if let Some(rest) = msg.strip_prefix(mlx_clairic::errors::QUEUE_FULL_PREFIX) {
        mlx_clairic::ClairicError::QueueFull(rest.to_string())
    } else if let Some(rest) = msg.strip_prefix(mlx_clairic::errors::INFERENCE_ERROR_PREFIX) {
        mlx_clairic::ClairicError::Inference(rest.to_string())
    } else {
        mlx_clairic::ClairicError::Message(msg)
    }
}

/// Pre-rendered RGB PDF page supplied from the host PDF renderer.
#[napi(object)]
pub struct RenderedPdfPage {
    pub rgb: Buffer,
    pub width: u32,
    pub height: u32,
    pub document_index: u32,
    pub page_index: u32,
}

/// Cooperative cancellation token for an in-flight Clairic extraction request.
#[napi]
pub struct ClairicCancellation {
    pub(crate) flag: Arc<AtomicBool>,
}

#[napi]
impl ClairicCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }
}

impl Default for ClairicCancellation {
    fn default() -> Self {
        Self::new()
    }
}

/// Native Clairic structured-extraction model sharing Qwen3.5's dedicated model thread.
#[napi]
pub struct ClairicModel {
    backbone: Qwen3_5Model,
    profile: ClairicProfile,
}

#[napi]
impl ClairicModel {
    /// Load a Clairic checkpoint directory or `.gguf` file (`clair_head_config.json`,
    /// `clair_head.safetensors`, `tokenizer.json`, and Qwen3.5-VL backbone weights).
    #[napi]
    pub async fn load(model_path: String) -> Result<Self> {
        let source =
            CheckpointSource::resolve(Path::new(&model_path), NativeGgufFamily::Qwen35).await?;
        match persistence::load_clairic_with_source(&source).await {
            Ok(backbone) => {
                let profile = backbone.clairic_profile().ok_or_else(|| {
                    to_napi_err(errors::inference_error(
                        "Clairic checkpoint did not initialize a profile",
                    ))
                })?;
                Ok(Self { backbone, profile })
            }
            Err(error) => {
                source.release_lease();
                Err(error)
            }
        }
    }

    /// Execute a structured Clairic extraction request given raw JSON (preserving key order
    /// and numeric spelling) and optional pre-rendered PDF pages.
    #[napi]
    pub async fn extract_json(
        &self,
        request: String,
        rendered_pdf_pages: Option<Vec<RenderedPdfPage>>,
        cancellation: Option<&ClairicCancellation>,
    ) -> Result<String> {
        if request.len() > orchestrator::MAX_REQUEST_BYTES {
            return Err(to_napi_err(errors::invalid_request(
                "request exceeds 16 MiB",
            )));
        }
        let pages: Vec<RenderedPdfPageData> = rendered_pdf_pages
            .unwrap_or_default()
            .into_iter()
            .map(|p| RenderedPdfPageData {
                rgb: p.rgb.to_vec(),
                width: p.width,
                height: p.height,
                document_index: p.document_index as usize,
                page_index: p.page_index as usize,
            })
            .collect();
        let cancelled = cancellation.map(|c| c.flag.clone());
        self.backbone
            .run_clairic(move |inner| {
                ar_engine::extract_json_on_inner(inner, &request, pages, cancelled.as_ref())
            })
            .await
    }

    /// Public model identifier (`"clairic-flash"` or `"clairic-27b"`).
    #[napi(getter)]
    pub fn model_id(&self) -> String {
        self.profile.public_model_id().to_string()
    }

    /// Maximum input prompt context length in tokens (`65_273`).
    #[napi(getter)]
    pub fn max_length(&self) -> u32 {
        settings::MAX_LENGTH as u32
    }
}
