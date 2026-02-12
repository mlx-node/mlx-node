//! Text Recognition Configuration

use serde::Deserialize;

use crate::models::pp_doclayout_v3::config::HGNetV2Config;

/// Configuration for the text recognition model.
#[derive(Debug, Clone, Deserialize)]
pub struct TextRecConfig {
    /// HGNetV2 backbone configuration
    #[serde(default = "default_backbone_config")]
    pub backbone: HGNetV2Config,

    /// Batch normalization epsilon
    #[serde(default = "default_bn_eps")]
    pub batch_norm_eps: f64,

    /// SVTR neck hidden dimension (default: 120 from PP-OCRv5_server_rec.yml)
    #[serde(default = "default_svtr_hidden_dim")]
    pub svtr_hidden_dim: i32,

    /// Number of SVTR transformer encoder layers
    #[serde(default = "default_svtr_num_layers")]
    pub svtr_num_layers: i32,

    /// Number of attention heads in SVTR
    #[serde(default = "default_svtr_num_heads")]
    pub svtr_num_heads: i32,

    /// FFN expansion ratio in SVTR (default: 2.0 for EncoderWithSVTR)
    #[serde(default = "default_svtr_ffn_ratio")]
    pub svtr_ffn_ratio: f64,

    /// SVTR output dimensionality (default: 120 from PP-OCRv5_server_rec.yml)
    #[serde(default = "default_svtr_output_dims")]
    pub svtr_output_dims: i32,

    /// Number of output classes (characters + blank)
    #[serde(default = "default_num_classes")]
    pub num_classes: i32,

    /// Input image height
    #[serde(default = "default_input_height")]
    pub input_height: i32,

    /// Input image max width
    #[serde(default = "default_input_max_width")]
    pub input_max_width: i32,
}

impl Default for TextRecConfig {
    fn default() -> Self {
        Self {
            backbone: default_backbone_config(),
            batch_norm_eps: default_bn_eps(),
            svtr_hidden_dim: default_svtr_hidden_dim(),
            svtr_num_layers: default_svtr_num_layers(),
            svtr_num_heads: default_svtr_num_heads(),
            svtr_ffn_ratio: default_svtr_ffn_ratio(),
            svtr_output_dims: default_svtr_output_dims(),
            num_classes: default_num_classes(),
            input_height: default_input_height(),
            input_max_width: default_input_max_width(),
        }
    }
}

/// PPHGNetV2_B4 backbone config for text recognition.
/// Matches PaddleOCR's PPHGNetV2_B4 with text_rec=True.
/// Uses asymmetric strides to preserve height for CTC: [(2,1),(1,2),(2,1),(2,1)].
/// stem3 uses stride=1 (not 2) when text_rec=True.
/// Reference: PaddleOCR/ppocr/modeling/backbones/rec_pphgnetv2.py
fn default_backbone_config() -> HGNetV2Config {
    HGNetV2Config {
        model_type: "hgnet_v2".to_string(),
        num_channels: 3,
        embedding_size: 48,
        depths: vec![1, 1, 3, 1],
        hidden_sizes: vec![128, 512, 1024, 2048],
        hidden_act: "relu".to_string(),
        stem_channels: vec![3, 32, 48],
        stage_in_channels: vec![48, 128, 512, 1024],
        stage_mid_channels: vec![48, 96, 192, 384],
        stage_out_channels: vec![128, 512, 1024, 2048],
        stage_num_blocks: vec![1, 1, 3, 1],
        stage_downsample: vec![true, true, true, true],
        stage_light_block: vec![false, false, true, true],
        stage_kernel_size: vec![3, 3, 5, 5],
        stage_numb_of_layers: vec![6, 6, 6, 6],
        use_learnable_affine_block: false,
        // Asymmetric strides for text recognition. At input height=48:
        // stem1 (2,2) → 24, stem3 (1,1) → 24, stage1 (2,1) → 12,
        // stage2 (1,2) → 12, stage3 (2,1) → 6, stage4 (2,1) → 3.
        // Final H=3 → avg_pool2d([3,2]) → H=1 → SVTR neck → CTC.
        stage_strides: vec![(2, 1), (1, 2), (2, 1), (2, 1)],
        // stem3 stride is (1,1) when text_rec=True
        stem3_stride: (1, 1),
        out_features: vec!["stage4".to_string()],
        out_indices: None,
        initializer_range: 0.02,
    }
}

fn default_bn_eps() -> f64 {
    1e-5
}
fn default_svtr_hidden_dim() -> i32 {
    120
}
fn default_svtr_num_layers() -> i32 {
    2
}
fn default_svtr_num_heads() -> i32 {
    8
}
fn default_svtr_ffn_ratio() -> f64 {
    2.0
}
fn default_svtr_output_dims() -> i32 {
    120
}
fn default_num_classes() -> i32 {
    6625
}
fn default_input_height() -> i32 {
    48
}
fn default_input_max_width() -> i32 {
    960
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_text_rec_config() {
        let config = TextRecConfig::default();
        // B4 rec backbone config
        assert_eq!(config.backbone.stem_channels, vec![3, 32, 48]);
        assert_eq!(config.backbone.stage_in_channels, vec![48, 128, 512, 1024]);
        assert_eq!(config.backbone.stage_mid_channels, vec![48, 96, 192, 384]);
        assert_eq!(
            config.backbone.stage_out_channels,
            vec![128, 512, 1024, 2048]
        );
        assert_eq!(config.backbone.stage_num_blocks, vec![1, 1, 3, 1]);
        assert_eq!(config.backbone.stage_numb_of_layers, vec![6, 6, 6, 6]);
        assert_eq!(
            config.backbone.stage_downsample,
            vec![true, true, true, true]
        );
        assert_eq!(
            config.backbone.stage_light_block,
            vec![false, false, true, true]
        );
        assert_eq!(config.backbone.stage_kernel_size, vec![3, 3, 5, 5]);
        assert!(!config.backbone.use_learnable_affine_block);
        // SVTR config
        assert_eq!(config.svtr_hidden_dim, 120);
        assert_eq!(config.svtr_num_layers, 2);
        assert_eq!(config.svtr_output_dims, 120);
        assert_eq!(config.num_classes, 6625);
        assert_eq!(config.input_height, 48);
        // stem3 stride is (1,1) for text_rec=True
        assert_eq!(config.backbone.stem3_stride, (1, 1));
        // Asymmetric strides for text rec
        assert_eq!(
            config.backbone.stage_strides,
            vec![(2, 1), (1, 2), (2, 1), (2, 1)]
        );
    }
}
