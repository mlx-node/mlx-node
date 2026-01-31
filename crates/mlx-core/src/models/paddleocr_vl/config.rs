/**
 * PaddleOCR-VL Configuration
 *
 * Model configuration for PaddleOCR-VL models.
 */
use napi_derive::napi;

/// Vision encoder configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_channels: i32,
    pub image_size: i32,
    pub patch_size: i32,
    pub hidden_act: String,
    pub layer_norm_eps: f64,
    pub attention_dropout: f64,
    pub spatial_merge_size: i32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            model_type: "paddleocr_vl".to_string(),
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 384,
            patch_size: 14,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            spatial_merge_size: 2,
        }
    }
}

/// Language model (text decoder) configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f64,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f64,
    pub rope_traditional: bool,
    pub use_bias: bool,
    pub head_dim: i32,
    /// Multimodal RoPE sections: [temporal, height, width]
    /// These define how the head_dim is split for 3D position encoding
    pub mrope_section: Vec<i32>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            model_type: "paddleocr_vl".to_string(),
            hidden_size: 1024,
            num_hidden_layers: 18,
            intermediate_size: 3072,
            num_attention_heads: 16,
            rms_norm_eps: 1e-5,
            vocab_size: 103424,
            num_key_value_heads: 2,
            max_position_embeddings: 131072,
            rope_theta: 500000.0,
            rope_traditional: false,
            use_bias: false,
            head_dim: 128,
            mrope_section: vec![16, 24, 24],
        }
    }
}

/// Full model configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub model_type: String,
    pub ignore_index: i32,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub eos_token_id: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            model_type: "paddleocr_vl".to_string(),
            ignore_index: -100,
            image_token_id: 100295,
            video_token_id: 100296,
            vision_start_token_id: 101305,
            vision_end_token_id: 101306,
            eos_token_id: 2,
        }
    }
}

impl ModelConfig {
    /// Create a new default PaddleOCR-VL 1.5 configuration
    pub fn paddleocr_vl_1_5() -> Self {
        Self::default()
    }

    /// Get the head dimension for attention
    pub fn head_dim(&self) -> i32 {
        self.text_config.head_dim
    }

    /// Get the number of query heads
    pub fn n_heads(&self) -> i32 {
        self.text_config.num_attention_heads
    }

    /// Get the number of key-value heads (for GQA)
    pub fn n_kv_heads(&self) -> i32 {
        self.text_config.num_key_value_heads
    }
}

/// Create a default PaddleOCR-VL 1.5 configuration (JS factory function)
#[napi]
pub fn create_paddleocr_vl_config() -> ModelConfig {
    ModelConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.model_type, "paddleocr_vl");
        assert_eq!(config.image_token_id, 100295);
        assert_eq!(config.text_config.hidden_size, 1024);
        assert_eq!(config.vision_config.hidden_size, 1152);
    }

    #[test]
    fn test_vision_config_defaults() {
        let config = VisionConfig::default();
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.num_hidden_layers, 27);
        assert_eq!(config.spatial_merge_size, 2);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4304);
        assert_eq!(config.image_size, 384);
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.layer_norm_eps, 1e-6);
    }

    #[test]
    fn test_text_config_defaults() {
        let config = TextConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 18);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.vocab_size, 103424);
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.rope_theta, 500000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);
    }

    #[test]
    fn test_mrope_section() {
        let config = TextConfig::default();
        // mrope_section should sum to head_dim/2 * 2 = head_dim
        // [16, 24, 24] * 2 = [32, 48, 48] -> cumsum = [32, 80, 128]
        assert_eq!(config.mrope_section, vec![16, 24, 24]);

        let total: i32 = config.mrope_section.iter().map(|&x| x * 2).sum();
        assert_eq!(total, config.head_dim);
    }

    #[test]
    fn test_model_config_special_tokens() {
        let config = ModelConfig::default();
        assert_eq!(config.image_token_id, 100295);
        assert_eq!(config.video_token_id, 100296);
        assert_eq!(config.vision_start_token_id, 101305);
        assert_eq!(config.vision_end_token_id, 101306);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.ignore_index, -100);
    }

    #[test]
    fn test_model_config_helpers() {
        let config = ModelConfig::default();
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.n_heads(), 16);
        assert_eq!(config.n_kv_heads(), 2);
    }

    #[test]
    fn test_factory_function() {
        let config = create_paddleocr_vl_config();
        assert_eq!(config.model_type, "paddleocr_vl");
        assert_eq!(config.vision_config.hidden_size, 1152);
        assert_eq!(config.text_config.hidden_size, 1024);
    }
}
