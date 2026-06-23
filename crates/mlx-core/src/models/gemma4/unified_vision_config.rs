/// Encoder-free vision configuration for the Gemma 4 unified multimodal model.
///
/// Parsed from the `vision_config` sub-dict of a `gemma4_unified` checkpoint
/// (`model_type == "gemma4_unified_vision"`). This is a different shape from the
/// SigLIP-style [`super::vision_config::Gemma4VisionConfig`] used by the dense
/// gemma4 family: the unified vision path has no transformer encoder, only a
/// patch embedder (LayerNorm + Linear + 2D positional embedding) feeding the
/// multimodal projection.
#[napi_derive::napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnifiedVisionConfig {
    /// Pixel side length of a single image patch (48 = patch_size 16 × pooling 3).
    pub model_patch_size: i32,
    /// Embedding width inside the vision embedder (3840, == text hidden_size).
    pub mm_embed_dim: i32,
    /// Number of rows in the 2D positional-embedding table (1120).
    pub mm_posemb_size: i32,
    /// Maximum soft tokens (patches) per image after resize (280).
    pub num_soft_tokens: i32,
    /// Output projection width of `embed_vision` (3840, == text hidden_size).
    pub output_proj_dims: i32,
    /// Pixel-grid patch size used by the resize math (16).
    pub patch_size: i32,
    /// Pooling kernel size used by the resize math (3).
    pub pooling_kernel_size: i32,
    /// Epsilon for the embedder LayerNorms and the projection RMSNorm.
    pub rms_norm_eps: f64,
}

impl UnifiedVisionConfig {
    /// Parse from the `vision_config` sub-dict of a unified checkpoint's
    /// config.json. Defaults match the 12B `gemma4_unified` release.
    pub fn from_json(vision_cfg: &serde_json::Value) -> Self {
        let get_i32 = |key: &str, default: i32| -> i32 {
            vision_cfg
                .get(key)
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            vision_cfg
                .get(key)
                .and_then(|v| v.as_f64())
                .unwrap_or(default)
        };

        Self {
            model_patch_size: get_i32("model_patch_size", 48),
            mm_embed_dim: get_i32("mm_embed_dim", 3840),
            mm_posemb_size: get_i32("mm_posemb_size", 1120),
            num_soft_tokens: get_i32("num_soft_tokens", 280),
            output_proj_dims: get_i32("output_proj_dims", 3840),
            patch_size: get_i32("patch_size", 16),
            pooling_kernel_size: get_i32("pooling_kernel_size", 3),
            rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        }
    }
}
