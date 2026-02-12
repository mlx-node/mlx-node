//! Text Detection Configuration
//!
//! Configuration for the DBNet text detection model with PPHGNetV2_B4 backbone.

use serde::Deserialize;

use crate::models::pp_doclayout_v3::config::HGNetV2Config;

/// Configuration for the text detection model.
#[derive(Debug, Clone, Deserialize)]
pub struct TextDetConfig {
    /// HGNetV2 backbone configuration
    #[serde(default = "default_backbone_config")]
    pub backbone: HGNetV2Config,

    /// Batch normalization epsilon
    #[serde(default = "default_bn_eps")]
    pub batch_norm_eps: f64,

    /// LKPAN output channels
    #[serde(default = "default_lkpan_out_channels")]
    pub lkpan_out_channels: i32,

    /// Large kernel size for LKPAN depthwise convolutions
    #[serde(default = "default_large_kernel_size")]
    pub large_kernel_size: i32,

    /// Activation function
    #[serde(default = "default_activation")]
    pub activation: String,

    /// DBHead inner channels (in_channels // 4 of LKPAN out)
    #[serde(default = "default_db_inner_channels")]
    pub db_inner_channels: i32,

    /// PFHeadLocal k value (for differentiable binarization step function)
    #[serde(default = "default_pf_head_k")]
    pub pf_head_k: i32,

    /// PFHeadLocal mode: "large" or "small" — controls LocalModule mid_channels
    #[serde(default = "default_pf_head_mode")]
    pub pf_head_mode: String,

    /// Detection threshold for binarization
    #[serde(default = "default_det_threshold")]
    pub det_threshold: f64,

    /// Box threshold (mean score inside box)
    #[serde(default = "default_box_threshold")]
    pub box_threshold: f64,

    /// Unclip ratio for box expansion
    #[serde(default = "default_unclip_ratio")]
    pub unclip_ratio: f64,

    /// Maximum number of candidates
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,

    /// Minimum box side length
    #[serde(default = "default_min_size")]
    pub min_size: f64,
}

impl Default for TextDetConfig {
    fn default() -> Self {
        Self {
            backbone: default_backbone_config(),
            batch_norm_eps: default_bn_eps(),
            lkpan_out_channels: default_lkpan_out_channels(),
            large_kernel_size: default_large_kernel_size(),
            activation: default_activation(),
            db_inner_channels: default_db_inner_channels(),
            pf_head_k: default_pf_head_k(),
            pf_head_mode: default_pf_head_mode(),
            det_threshold: default_det_threshold(),
            box_threshold: default_box_threshold(),
            unclip_ratio: default_unclip_ratio(),
            max_candidates: default_max_candidates(),
            min_size: default_min_size(),
        }
    }
}

/// PPHGNetV2_B4 backbone config for text detection.
/// Matches PaddleOCR's `PPHGNetV2_B4(det=True)` with `stage_config_det`.
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
        stage_downsample: vec![false, true, true, true],
        stage_light_block: vec![false, false, true, true],
        stage_kernel_size: vec![3, 3, 5, 5],
        stage_numb_of_layers: vec![6, 6, 6, 6],
        use_learnable_affine_block: false,
        stage_strides: vec![(2, 2), (2, 2), (2, 2), (2, 2)],
        stem3_stride: (2, 2),
        out_features: vec![
            "stage1".to_string(),
            "stage2".to_string(),
            "stage3".to_string(),
            "stage4".to_string(),
        ],
        out_indices: None,
        initializer_range: 0.02,
    }
}

fn default_bn_eps() -> f64 {
    1e-5
}
fn default_lkpan_out_channels() -> i32 {
    256
}
fn default_large_kernel_size() -> i32 {
    9
}
fn default_activation() -> String {
    "relu".to_string()
}
fn default_db_inner_channels() -> i32 {
    64
}
fn default_pf_head_k() -> i32 {
    50
}
fn default_pf_head_mode() -> String {
    "large".to_string()
}
fn default_det_threshold() -> f64 {
    0.3
}
fn default_box_threshold() -> f64 {
    0.6
}
fn default_unclip_ratio() -> f64 {
    1.5
}
fn default_max_candidates() -> usize {
    1000
}
fn default_min_size() -> f64 {
    3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_text_det_config() {
        let config = TextDetConfig::default();
        assert_eq!(config.backbone.stem_channels, vec![3, 32, 48]);
        assert_eq!(
            config.backbone.stage_out_channels,
            vec![128, 512, 1024, 2048]
        );
        assert_eq!(config.backbone.stage_numb_of_layers, vec![6, 6, 6, 6]);
        assert_eq!(config.backbone.embedding_size, 48);
        assert_eq!(config.lkpan_out_channels, 256);
        assert_eq!(config.large_kernel_size, 9);
        assert_eq!(config.db_inner_channels, 64);
        assert!((config.det_threshold - 0.3).abs() < 1e-10);
        assert!((config.box_threshold - 0.6).abs() < 1e-10);
    }
}
