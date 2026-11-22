//! Clairic runtime constants and settings matching upstream `clairic_runtime/{__init__,settings,embedded,mlx_engine,backend}.py`.

use super::errors::invalid_request;
use napi::Result;

/// Full context window (input + headroom) in tokens (`CONTEXT_WINDOW` / `MAX_PASS_TOKENS`).
pub const MAX_CONTEXT_TOKENS: usize = 65_536;
/// Maximum autoregressive output tokens per field (`MAX_OUTPUT_TOKENS`).
pub const MAX_OUTPUT_TOKENS: usize = 256;
/// Reserved headroom tokens (`CONTEXT_HEADROOM_TOKENS = MAX_OUTPUT_TOKENS + 7`).
pub const CONTEXT_HEADROOM_TOKENS: usize = MAX_OUTPUT_TOKENS + 7;
/// Maximum input prompt length in tokens (`DEFAULT_MAX_LENGTH = 65_536 - 263 = 65_273`).
pub const MAX_LENGTH: usize = MAX_CONTEXT_TOKENS - CONTEXT_HEADROOM_TOKENS;
/// Backbone prefill segment token budget (`SEGMENT_TOKENS` in `mlx_engine.py`).
pub const SEGMENT_TOKENS: usize = 1024;

/// Maximum embedded images per request (`embedded.MAX_IMAGES`).
pub const MAX_IMAGES: usize = 4;
/// Maximum decoded byte size of a single image (`embedded.MAX_IMAGE_BYTES`).
pub const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum aggregate decoded byte size across all images in a request (`embedded.MAX_IMAGES_BYTES`).
pub const MAX_IMAGES_BYTES: usize = 8 * 1024 * 1024;
/// Maximum raster pixel count accepted by image validation (`embedded.normalize_embedded_image`).
pub const MAX_PIXELS: usize = 16_000_000;
/// Maximum pixel budget for vision supplements (`po_image_supplement.MAX_PIXELS`).
pub const VISION_MAX_PIXELS: usize = 1_600_000;
/// Maximum pixel budget when spelling identifiers across many images (`identifier_spelling.MANY_IMAGE_MAX_PIXELS`).
pub const MANY_IMAGE_MAX_PIXELS: usize = 800_000;

/// Maximum concurrent requests enforced by upstream `RuntimeSettings`.
pub const MAX_CONCURRENCY: usize = 1;
/// Default serving API version (`RuntimeSettings.version`).
pub const DEFAULT_VERSION: &str = "1";
/// Generic public model identifier (`api.MODEL_ID`).
pub const PUBLIC_MODEL_ID: &str = "clairic";

/// Request timeout in seconds (`ClairicFlashModel.REQUEST_TIMEOUT_SECONDS`).
pub const REQUEST_TIMEOUT_SECONDS: u64 = 600;
/// Date supplement timeout in seconds (`ClairicFlashModel.DATE_SUPPLEMENT_TIMEOUT_SECONDS`).
pub const DATE_SUPPLEMENT_TIMEOUT_SECONDS: u64 = 600;
/// Receipt supplement timeout in seconds (`ClairicFlashModel.RECEIPT_SUPPLEMENT_TIMEOUT_SECONDS`).
pub const RECEIPT_SUPPLEMENT_TIMEOUT_SECONDS: u64 = 600;

/// Configuration for a local Clairic model instance (port of `settings.RuntimeSettings`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClairicSettings {
    pub model_path: String,
    pub model_id: String,
    pub max_length: usize,
    pub max_concurrency: usize,
    pub version: String,
    pub model_aliases: Vec<String>,
}

impl ClairicSettings {
    pub fn new(model_path: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: model_id.into(),
            max_length: MAX_LENGTH,
            max_concurrency: MAX_CONCURRENCY,
            version: DEFAULT_VERSION.to_string(),
            model_aliases: Vec::new(),
        }
    }

    /// Validate settings against upstream `RuntimeSettings.__post_init__` invariants.
    pub fn validate(&self) -> Result<()> {
        if self.max_length == 0 || self.max_length > MAX_LENGTH {
            return Err(invalid_request(format!(
                "max_length must be an integer from 1 to {MAX_LENGTH}"
            )));
        }
        if self.max_concurrency != MAX_CONCURRENCY {
            return Err(invalid_request("max_concurrency must be 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_upstream() {
        assert_eq!(MAX_CONTEXT_TOKENS, 65_536);
        assert_eq!(MAX_OUTPUT_TOKENS, 256);
        assert_eq!(CONTEXT_HEADROOM_TOKENS, 263);
        assert_eq!(MAX_LENGTH, 65_273);
        assert_eq!(SEGMENT_TOKENS, 1024);
        assert_eq!(MAX_IMAGES, 4);
        assert_eq!(MAX_IMAGE_BYTES, 4 * 1024 * 1024);
        assert_eq!(MAX_IMAGES_BYTES, 8 * 1024 * 1024);
        assert_eq!(MAX_PIXELS, 16_000_000);
        assert_eq!(VISION_MAX_PIXELS, 1_600_000);
        assert_eq!(MANY_IMAGE_MAX_PIXELS, 800_000);
    }

    #[test]
    fn settings_validation() {
        let mut settings = ClairicSettings::new("/tmp/model", "clairic-flash");
        assert!(settings.validate().is_ok());

        settings.max_length = 0;
        assert!(settings.validate().is_err());

        settings.max_length = MAX_LENGTH + 1;
        assert!(settings.validate().is_err());

        settings.max_length = MAX_LENGTH;
        settings.max_concurrency = 2;
        assert!(settings.validate().is_err());
    }
}
