/**
 * Qianfan-OCR Configuration
 *
 * Model configuration for Qianfan-OCR (InternVL) models.
 */
use napi_derive::napi;

/// InternViT vision encoder configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct InternVisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_channels: i32,
    pub image_size: i32,
    pub patch_size: i32,
    pub layer_norm_eps: f64,
    pub qkv_bias: bool,
    /// Drop path rate (inference only, always 0)
    pub drop_path_rate: f64,
}

impl Default for InternVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 448,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            qkv_bias: true,
            drop_path_rate: 0.0,
        }
    }
}

/// Qwen3 language model configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3LMConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f64,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f64,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen3LMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            vocab_size: 153678,
            max_position_embeddings: 32768,
            rope_theta: 5_000_000.0,
            use_qk_norm: true,
            tie_word_embeddings: false,
        }
    }
}

/// Full Qianfan-OCR model configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct QianfanOCRConfig {
    pub vision_config: InternVisionConfig,
    pub llm_config: Qwen3LMConfig,
    pub model_type: String,
    pub img_context_token_id: i32,
    /// `<img>` token ID
    pub img_start_token_id: i32,
    /// `</img>` token ID
    pub img_end_token_id: i32,
    /// `<|im_end|>` token ID
    pub eos_token_id: i32,
    /// Which vision encoder layer to extract features from
    pub select_layer: i32,
    /// Pixel shuffle version
    pub ps_version: String,
    pub downsample_ratio: f64,
    pub dynamic_image_size: bool,
    pub use_thumbnail: bool,
    pub max_dynamic_patch: i32,
    pub min_dynamic_patch: i32,
}

impl Default for QianfanOCRConfig {
    fn default() -> Self {
        Self {
            vision_config: InternVisionConfig::default(),
            llm_config: Qwen3LMConfig::default(),
            model_type: "internvl_chat".to_string(),
            img_context_token_id: 151671,
            img_start_token_id: 151669,
            img_end_token_id: 151670,
            eos_token_id: 151645,
            select_layer: -1,
            ps_version: "v2".to_string(),
            downsample_ratio: 0.5,
            dynamic_image_size: true,
            use_thumbnail: true,
            max_dynamic_patch: 12,
            min_dynamic_patch: 1,
        }
    }
}

impl QianfanOCRConfig {
    /// Compute the number of image tokens per patch.
    ///
    /// Formula: ((image_size / patch_size)^2) * (downsample_ratio^2)
    /// With defaults: ((448/14)^2) * (0.5^2) = 1024 * 0.25 = 256
    pub fn num_image_token(&self) -> i32 {
        let grid = self.vision_config.image_size / self.vision_config.patch_size;
        let tokens = grid * grid;
        (tokens as f64 * self.downsample_ratio * self.downsample_ratio) as i32
    }

    /// Get the head dimension from the language model config
    pub fn head_dim(&self) -> i32 {
        self.llm_config.head_dim
    }
}

/// Create a default Qianfan-OCR configuration (JS factory function)
#[napi]
pub fn create_qianfan_ocr_config() -> QianfanOCRConfig {
    QianfanOCRConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QianfanOCRConfig::default();
        assert_eq!(config.model_type, "internvl_chat");
        assert_eq!(config.img_context_token_id, 151671);
        assert_eq!(config.llm_config.hidden_size, 2560);
        assert_eq!(config.vision_config.hidden_size, 1024);
    }

    #[test]
    fn test_vision_config_defaults() {
        let config = InternVisionConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.intermediate_size, 4096);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.image_size, 448);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.layer_norm_eps, 1e-6);
        assert!(config.qkv_bias);
        assert_eq!(config.drop_path_rate, 0.0);
    }

    #[test]
    fn test_qwen3_lm_config_defaults() {
        let config = Qwen3LMConfig::default();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.intermediate_size, 9728);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.vocab_size, 153678);
        assert_eq!(config.max_position_embeddings, 32768);
        assert_eq!(config.rope_theta, 5_000_000.0);
        assert!(config.use_qk_norm);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_model_config_special_tokens() {
        let config = QianfanOCRConfig::default();
        assert_eq!(config.img_context_token_id, 151671);
        assert_eq!(config.img_start_token_id, 151669);
        assert_eq!(config.img_end_token_id, 151670);
        assert_eq!(config.eos_token_id, 151645);
    }

    #[test]
    fn test_model_config_image_settings() {
        let config = QianfanOCRConfig::default();
        assert_eq!(config.select_layer, -1);
        assert_eq!(config.ps_version, "v2");
        assert_eq!(config.downsample_ratio, 0.5);
        assert!(config.dynamic_image_size);
        assert!(config.use_thumbnail);
        assert_eq!(config.max_dynamic_patch, 12);
        assert_eq!(config.min_dynamic_patch, 1);
    }

    #[test]
    fn test_num_image_token() {
        let config = QianfanOCRConfig::default();
        // ((448/14)^2) * (0.5^2) = (32^2) * 0.25 = 1024 * 0.25 = 256
        assert_eq!(config.num_image_token(), 256);
    }

    #[test]
    fn test_num_image_token_custom() {
        let mut config = QianfanOCRConfig::default();
        config.vision_config.image_size = 224;
        config.vision_config.patch_size = 14;
        config.downsample_ratio = 1.0;
        // ((224/14)^2) * (1.0^2) = (16^2) * 1.0 = 256
        assert_eq!(config.num_image_token(), 256);
    }

    #[test]
    fn test_head_dim() {
        let config = QianfanOCRConfig::default();
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_factory_function() {
        let config = create_qianfan_ocr_config();
        assert_eq!(config.model_type, "internvl_chat");
        assert_eq!(config.vision_config.hidden_size, 1024);
        assert_eq!(config.llm_config.hidden_size, 2560);
    }
}
