//! Clairic structured-extraction runtime modules and N-API bindings.

pub mod ar;
pub mod ar_engine;
pub mod completion;
pub mod contracts;
pub mod crops;
pub mod decimal_util;
pub mod embedded;
pub mod errors;
pub mod grounding;
pub mod head;
pub mod html_rows;
pub mod identifier_spelling;
pub mod image_rows;
pub mod json_validate;
pub mod lists;
pub mod media;
pub mod media_cache;
pub mod media_state_prompt;
pub mod orchestrator;
pub mod orientation;
pub mod printed_numbers;
pub mod printed_temporal;
pub mod profile;
pub mod receipt;
pub mod row_omission;
pub mod schema;
pub mod scopes;
pub mod settings;
pub mod source_dates;
pub mod source_rows;
pub mod table_rows;
pub mod typed_decode;
pub mod typed_prompt;

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

use self::embedded::RenderedPdfPageData;
use self::profile::ClairicProfile;

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
                    errors::inference_error("Clairic checkpoint did not initialize a profile")
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
            return Err(errors::invalid_request("request exceeds 16 MiB"));
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
                orchestrator::extract_json_on_inner(inner, &request, pages, cancelled.as_ref())
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
