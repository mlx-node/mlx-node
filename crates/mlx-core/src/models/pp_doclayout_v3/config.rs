//! PP-DocLayoutV3 Configuration
//!
//! Model configuration for PP-DocLayoutV3 document layout analysis models.
//! Based on the HuggingFace Transformers PPDocLayoutV3Config and HGNetV2Config.

use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// HGNetV2 Backbone Configuration
// ============================================================================

/// Configuration for the HGNetV2 backbone used in PP-DocLayoutV3.
///
/// The HGNetV2 architecture consists of:
/// - A stem (embedding) layer that downsamples the input
/// - 4 stages of HGBlocks with increasing channel dimensions
///
/// Default values correspond to the HGNetV2-L architecture.
#[derive(Debug, Clone, Deserialize)]
pub struct HGNetV2Config {
    /// Model type identifier
    #[serde(default = "default_hgnet_v2_model_type")]
    pub model_type: String,

    /// Number of input image channels (default: 3 for RGB)
    #[serde(default = "default_num_channels")]
    pub num_channels: i32,

    /// Dimensionality of the embedding (stem output) layer
    #[serde(default = "default_embedding_size")]
    pub embedding_size: i32,

    /// Depth (number of layers) for each stage
    #[serde(default = "default_depths")]
    pub depths: Vec<i32>,

    /// Hidden sizes (output channels) at each stage
    #[serde(default = "default_hidden_sizes")]
    pub hidden_sizes: Vec<i32>,

    /// Activation function: "relu" or "silu"
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Channel dimensions for the stem layers: [input, intermediate, output]
    #[serde(default = "default_stem_channels")]
    pub stem_channels: Vec<i32>,

    /// Input channel dimensions for each stage
    #[serde(default = "default_stage_in_channels")]
    pub stage_in_channels: Vec<i32>,

    /// Mid-channel dimensions for each stage
    #[serde(default = "default_stage_mid_channels")]
    pub stage_mid_channels: Vec<i32>,

    /// Output channel dimensions for each stage
    #[serde(default = "default_stage_out_channels")]
    pub stage_out_channels: Vec<i32>,

    /// Number of HGBlocks in each stage
    #[serde(default = "default_stage_num_blocks")]
    pub stage_num_blocks: Vec<i32>,

    /// Whether to downsample at each stage
    #[serde(default = "default_stage_downsample")]
    pub stage_downsample: Vec<bool>,

    /// Whether to use light (depthwise separable) blocks in each stage
    #[serde(default = "default_stage_light_block")]
    pub stage_light_block: Vec<bool>,

    /// Kernel sizes for convolutions in each stage
    #[serde(default = "default_stage_kernel_size")]
    pub stage_kernel_size: Vec<i32>,

    /// Number of conv layers within each HGBlock per stage
    #[serde(default = "default_stage_numb_of_layers")]
    pub stage_numb_of_layers: Vec<i32>,

    /// Whether to use Learnable Affine Blocks after activations
    #[serde(default)]
    pub use_learnable_affine_block: bool,

    /// Per-stage downsample strides as (stride_h, stride_w).
    /// Default: all (2,2). Text recognition uses asymmetric strides like [(2,1),(1,2),(2,1),(2,1)].
    #[serde(default = "default_stage_strides")]
    pub stage_strides: Vec<(i32, i32)>,

    /// Stride for stem3 layer. Default: (2,2). Text recognition may use (2,1).
    #[serde(default = "default_stem3_stride")]
    pub stem3_stride: (i32, i32),

    /// Which stage outputs to return (e.g., ["stage1", "stage2", "stage3", "stage4"])
    #[serde(default = "default_out_features")]
    pub out_features: Vec<String>,

    /// Indices of output stages
    #[serde(default)]
    pub out_indices: Option<Vec<i32>>,

    /// Standard deviation for weight initialization
    #[serde(default = "default_initializer_range_backbone")]
    pub initializer_range: f64,
}

impl Default for HGNetV2Config {
    fn default() -> Self {
        Self {
            model_type: default_hgnet_v2_model_type(),
            num_channels: default_num_channels(),
            embedding_size: default_embedding_size(),
            depths: default_depths(),
            hidden_sizes: default_hidden_sizes(),
            hidden_act: default_hidden_act(),
            stem_channels: default_stem_channels(),
            stage_in_channels: default_stage_in_channels(),
            stage_mid_channels: default_stage_mid_channels(),
            stage_out_channels: default_stage_out_channels(),
            stage_num_blocks: default_stage_num_blocks(),
            stage_downsample: default_stage_downsample(),
            stage_light_block: default_stage_light_block(),
            stage_kernel_size: default_stage_kernel_size(),
            stage_numb_of_layers: default_stage_numb_of_layers(),
            use_learnable_affine_block: false,
            stage_strides: default_stage_strides(),
            stem3_stride: default_stem3_stride(),
            out_features: default_out_features(),
            out_indices: None,
            initializer_range: default_initializer_range_backbone(),
        }
    }
}

impl HGNetV2Config {
    /// Get the number of stages
    pub fn num_stages(&self) -> usize {
        self.stage_in_channels.len()
    }

    /// Get the output channel sizes for the stages that are in out_features.
    /// Returns all 4 stage output channels (the backbone always runs all 4 stages
    /// and returns the requested features).
    pub fn output_channels(&self) -> Vec<i32> {
        self.out_features
            .iter()
            .filter_map(|name| {
                if let Some(idx_str) = name.strip_prefix("stage")
                    && let Ok(idx) = idx_str.parse::<usize>()
                {
                    // stage1 = index 0, stage2 = index 1, etc.
                    return self.stage_out_channels.get(idx.saturating_sub(1)).copied();
                }
                None
            })
            .collect()
    }
}

// ============================================================================
// HGNetV2Config defaults (HGNetV2-L architecture)
// ============================================================================

fn default_hgnet_v2_model_type() -> String {
    "hgnet_v2".to_string()
}
fn default_num_channels() -> i32 {
    3
}
fn default_embedding_size() -> i32 {
    48 // stem_channels[2] for B4/L variant
}
fn default_depths() -> Vec<i32> {
    vec![3, 4, 6, 3]
}
fn default_hidden_sizes() -> Vec<i32> {
    vec![256, 512, 1024, 2048]
}
fn default_hidden_act() -> String {
    "relu".to_string()
}
fn default_stem_channels() -> Vec<i32> {
    vec![3, 32, 48]
}
fn default_stage_in_channels() -> Vec<i32> {
    vec![48, 128, 512, 1024]
}
fn default_stage_mid_channels() -> Vec<i32> {
    vec![48, 96, 192, 384]
}
fn default_stage_out_channels() -> Vec<i32> {
    vec![128, 512, 1024, 2048]
}
fn default_stage_num_blocks() -> Vec<i32> {
    vec![1, 1, 3, 1]
}
fn default_stage_downsample() -> Vec<bool> {
    vec![false, true, true, true]
}
fn default_stage_light_block() -> Vec<bool> {
    vec![false, false, true, true]
}
fn default_stage_kernel_size() -> Vec<i32> {
    vec![3, 3, 5, 5]
}
fn default_stage_numb_of_layers() -> Vec<i32> {
    vec![6, 6, 6, 6]
}
fn default_stage_strides() -> Vec<(i32, i32)> {
    vec![(2, 2), (2, 2), (2, 2), (2, 2)]
}
fn default_stem3_stride() -> (i32, i32) {
    (2, 2)
}
fn default_out_features() -> Vec<String> {
    vec![
        "stage1".to_string(),
        "stage2".to_string(),
        "stage3".to_string(),
        "stage4".to_string(),
    ]
}
fn default_initializer_range_backbone() -> f64 {
    0.02
}

// ============================================================================
// PP-DocLayoutV3 Full Model Configuration
// ============================================================================

/// Full configuration for the PP-DocLayoutV3 model.
///
/// This includes the backbone config plus all encoder/decoder parameters.
/// Corresponds to PPDocLayoutV3Config in HuggingFace Transformers.
#[derive(Debug, Clone, Deserialize)]
pub struct PPDocLayoutV3Config {
    /// Standard deviation for weight initialization
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,

    /// Bias initializer prior probability (None uses 1/(num_labels+1))
    #[serde(default)]
    pub initializer_bias_prior_prob: Option<f64>,

    /// Epsilon for layer normalization
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    /// Epsilon for batch normalization
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f64,

    // -- Backbone --
    /// HGNetV2 backbone configuration
    #[serde(default)]
    pub backbone_config: HGNetV2Config,

    /// Whether to freeze batch norms in the backbone
    #[serde(default = "default_true")]
    pub freeze_backbone_batch_norms: bool,

    // -- Encoder (PPDocLayoutV3HybridEncoder) --
    /// Hidden dimension of the encoder
    #[serde(default = "default_encoder_hidden_dim")]
    pub encoder_hidden_dim: i32,

    /// Input channel dimensions from backbone to encoder
    #[serde(default = "default_encoder_in_channels")]
    pub encoder_in_channels: Vec<i32>,

    /// Feature strides for each feature level
    #[serde(default = "default_feat_strides", alias = "feature_strides")]
    pub feat_strides: Vec<i32>,

    /// Number of transformer encoder layers
    #[serde(default = "default_one")]
    pub encoder_layers: i32,

    /// FFN dimension in the encoder
    #[serde(default = "default_ffn_dim")]
    pub encoder_ffn_dim: i32,

    /// Number of attention heads in the encoder
    #[serde(default = "default_eight")]
    pub encoder_attention_heads: i32,

    /// Dropout rate
    #[serde(default)]
    pub dropout: f64,

    /// Activation dropout rate
    #[serde(default)]
    pub activation_dropout: f64,

    /// Indices of projected encoder layers
    #[serde(default = "default_encode_proj_layers")]
    pub encode_proj_layers: Vec<i32>,

    /// Temperature for positional encoding
    #[serde(default = "default_positional_encoding_temperature")]
    pub positional_encoding_temperature: i32,

    /// Encoder activation function (e.g., "gelu")
    #[serde(default = "default_encoder_activation")]
    pub encoder_activation_function: String,

    /// General activation function (e.g., "silu")
    #[serde(default = "default_activation_function")]
    pub activation_function: String,

    /// Eval image size [height, width] for fixed positional embeddings
    #[serde(default)]
    pub eval_size: Option<Vec<i32>>,

    /// Whether to apply layer norm before attention
    #[serde(default)]
    pub normalize_before: bool,

    /// Expansion ratio for RepVGG/CSPRep blocks
    #[serde(default = "default_one_f64")]
    pub hidden_expansion: f64,

    /// Channels for mask feature FPN
    #[serde(default = "default_mask_feature_channels")]
    pub mask_feature_channels: Vec<i32>,

    /// Dimension of the x4 (stride-4) feature map
    #[serde(default = "default_x4_feat_dim")]
    pub x4_feat_dim: i32,

    // -- Decoder (PPDocLayoutV3Transformer) --
    /// Model dimension (used in decoder, queries, heads)
    #[serde(default = "default_d_model")]
    pub d_model: i32,

    /// Number of prototypes for mask prediction
    #[serde(default = "default_num_prototypes")]
    pub num_prototypes: i32,

    /// Label noise ratio for denoising training
    #[serde(default = "default_noise_ratio")]
    pub label_noise_ratio: f64,

    /// Box noise scale for denoising training
    #[serde(default = "default_noise_ratio")]
    pub box_noise_scale: f64,

    /// Whether to use mask-enhanced attention
    #[serde(default = "default_true")]
    pub mask_enhanced: bool,

    /// Number of object queries
    #[serde(default = "default_num_queries")]
    pub num_queries: i32,

    /// Decoder input channel dimensions
    #[serde(default = "default_decoder_in_channels")]
    pub decoder_in_channels: Vec<i32>,

    /// FFN dimension in the decoder
    #[serde(default = "default_ffn_dim")]
    pub decoder_ffn_dim: i32,

    /// Number of feature levels in the decoder
    #[serde(default = "default_num_feature_levels")]
    pub num_feature_levels: i32,

    /// Number of sampling points per attention head in the decoder
    #[serde(default = "default_decoder_n_points")]
    pub decoder_n_points: i32,

    /// Number of decoder layers
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: i32,

    /// Number of attention heads in the decoder
    #[serde(default = "default_eight")]
    pub decoder_attention_heads: i32,

    /// Decoder activation function (e.g., "relu")
    #[serde(default = "default_decoder_activation")]
    pub decoder_activation_function: String,

    /// Attention dropout rate
    #[serde(default)]
    pub attention_dropout: f64,

    /// Number of denoising queries
    #[serde(default = "default_num_denoising")]
    pub num_denoising: i32,

    /// Whether to learn initial query embeddings
    #[serde(default)]
    pub learn_initial_query: bool,

    /// Anchor image size for fixed anchors [height, width]
    #[serde(default)]
    pub anchor_image_size: Option<Vec<i32>>,

    /// Whether to disable custom CUDA kernels
    #[serde(default = "default_true")]
    pub disable_custom_kernels: bool,

    /// Global pointer head size
    #[serde(default = "default_global_pointer_head_size")]
    pub global_pointer_head_size: i32,

    /// Global pointer dropout value
    #[serde(default = "default_gp_dropout")]
    pub gp_dropout_value: f64,

    // -- Labels --
    /// Number of label classes (layout categories)
    #[serde(default = "default_num_labels")]
    pub num_labels: i32,

    /// Mapping from class ID to label name
    #[serde(default = "default_id2label")]
    pub id2label: HashMap<String, String>,

    /// Mapping from label name to class ID
    #[serde(default)]
    pub label2id: HashMap<String, i32>,
}

impl Default for PPDocLayoutV3Config {
    fn default() -> Self {
        Self {
            initializer_range: default_initializer_range(),
            initializer_bias_prior_prob: None,
            layer_norm_eps: default_layer_norm_eps(),
            batch_norm_eps: default_batch_norm_eps(),
            backbone_config: HGNetV2Config::default(),
            freeze_backbone_batch_norms: true,
            encoder_hidden_dim: default_encoder_hidden_dim(),
            encoder_in_channels: default_encoder_in_channels(),
            feat_strides: default_feat_strides(),
            encoder_layers: 1,
            encoder_ffn_dim: default_ffn_dim(),
            encoder_attention_heads: 8,
            dropout: 0.0,
            activation_dropout: 0.0,
            encode_proj_layers: default_encode_proj_layers(),
            positional_encoding_temperature: default_positional_encoding_temperature(),
            encoder_activation_function: default_encoder_activation(),
            activation_function: default_activation_function(),
            eval_size: None,
            normalize_before: false,
            hidden_expansion: 1.0,
            mask_feature_channels: default_mask_feature_channels(),
            x4_feat_dim: default_x4_feat_dim(),
            d_model: default_d_model(),
            num_prototypes: default_num_prototypes(),
            label_noise_ratio: default_noise_ratio(),
            box_noise_scale: default_noise_ratio(),
            mask_enhanced: true,
            num_queries: default_num_queries(),
            decoder_in_channels: default_decoder_in_channels(),
            decoder_ffn_dim: default_ffn_dim(),
            num_feature_levels: default_num_feature_levels(),
            decoder_n_points: default_decoder_n_points(),
            decoder_layers: default_decoder_layers(),
            decoder_attention_heads: 8,
            decoder_activation_function: default_decoder_activation(),
            attention_dropout: 0.0,
            num_denoising: default_num_denoising(),
            learn_initial_query: false,
            anchor_image_size: None,
            disable_custom_kernels: true,
            global_pointer_head_size: default_global_pointer_head_size(),
            gp_dropout_value: default_gp_dropout(),
            num_labels: default_num_labels(),
            id2label: default_id2label(),
            label2id: HashMap::new(),
        }
    }
}

impl PPDocLayoutV3Config {
    /// Get the intermediate channel sizes from the backbone (for stages in out_features)
    pub fn backbone_output_channels(&self) -> Vec<i32> {
        self.backbone_config.output_channels()
    }

    /// Fix up derived fields after deserialization.
    ///
    /// HuggingFace derives `num_labels` from `id2label` when not explicitly provided.
    /// The actual PP-DocLayoutV3 config.json omits `num_labels`, so we infer it here.
    pub fn fixup_after_load(&mut self) {
        if !self.id2label.is_empty() {
            let inferred = self.id2label.len() as i32;
            // Only override if num_labels was left at default and id2label disagrees
            if self.num_labels != inferred {
                self.num_labels = inferred;
            }
        }
    }
}

// ============================================================================
// PPDocLayoutV3Config defaults
// ============================================================================

fn default_initializer_range() -> f64 {
    0.01
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_batch_norm_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_encoder_hidden_dim() -> i32 {
    256
}
fn default_encoder_in_channels() -> Vec<i32> {
    vec![512, 1024, 2048]
}
fn default_feat_strides() -> Vec<i32> {
    vec![8, 16, 32]
}
fn default_one() -> i32 {
    1
}
fn default_ffn_dim() -> i32 {
    1024
}
fn default_eight() -> i32 {
    8
}
fn default_encode_proj_layers() -> Vec<i32> {
    vec![2]
}
fn default_positional_encoding_temperature() -> i32 {
    10000
}
fn default_encoder_activation() -> String {
    "gelu".to_string()
}
fn default_activation_function() -> String {
    "silu".to_string()
}
fn default_one_f64() -> f64 {
    1.0
}
fn default_mask_feature_channels() -> Vec<i32> {
    vec![64, 64]
}
fn default_x4_feat_dim() -> i32 {
    128
}
fn default_d_model() -> i32 {
    256
}
fn default_num_prototypes() -> i32 {
    32
}
fn default_noise_ratio() -> f64 {
    0.4
}
fn default_num_queries() -> i32 {
    300
}
fn default_decoder_in_channels() -> Vec<i32> {
    vec![256, 256, 256]
}
fn default_num_feature_levels() -> i32 {
    3
}
fn default_decoder_n_points() -> i32 {
    4
}
fn default_decoder_layers() -> i32 {
    6
}
fn default_decoder_activation() -> String {
    "relu".to_string()
}
fn default_num_denoising() -> i32 {
    100
}
fn default_global_pointer_head_size() -> i32 {
    64
}
fn default_gp_dropout() -> f64 {
    0.1
}
fn default_num_labels() -> i32 {
    25
}

fn default_id2label() -> HashMap<String, String> {
    // Default labels matching the PaddlePaddle/PP-DocLayoutV3_safetensors HuggingFace model.
    // At runtime, these are overridden by id2label from the model's config.json.
    let labels = [
        (0, "abstract"),
        (1, "algorithm"),
        (2, "aside_text"),
        (3, "chart"),
        (4, "content"),
        (5, "formula"),
        (6, "doc_title"),
        (7, "figure_title"),
        (8, "footer"),
        (9, "footer"),
        (10, "footnote"),
        (11, "formula_number"),
        (12, "header"),
        (13, "header"),
        (14, "image"),
        (15, "formula"),
        (16, "number"),
        (17, "paragraph_title"),
        (18, "reference"),
        (19, "reference_content"),
        (20, "seal"),
        (21, "table"),
        (22, "text"),
        (23, "text"),
        (24, "vision_footnote"),
    ];
    labels
        .iter()
        .map(|(id, label)| (id.to_string(), label.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hgnetv2_config() {
        let config = HGNetV2Config::default();
        assert_eq!(config.model_type, "hgnet_v2");
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.embedding_size, 48);
        assert_eq!(config.stem_channels, vec![3, 32, 48]);
        assert_eq!(config.stage_in_channels, vec![48, 128, 512, 1024]);
        assert_eq!(config.stage_mid_channels, vec![48, 96, 192, 384]);
        assert_eq!(config.stage_out_channels, vec![128, 512, 1024, 2048]);
        assert_eq!(config.stage_num_blocks, vec![1, 1, 3, 1]);
        assert_eq!(config.stage_downsample, vec![false, true, true, true]);
        assert_eq!(config.stage_light_block, vec![false, false, true, true]);
        assert_eq!(config.stage_kernel_size, vec![3, 3, 5, 5]);
        assert_eq!(config.stage_numb_of_layers, vec![6, 6, 6, 6]);
        assert!(!config.use_learnable_affine_block);
        assert_eq!(config.hidden_act, "relu");
        assert_eq!(config.num_stages(), 4);
    }

    #[test]
    fn test_hgnetv2_output_channels() {
        let config = HGNetV2Config::default();
        let channels = config.output_channels();
        assert_eq!(channels, vec![128, 512, 1024, 2048]);
    }

    #[test]
    fn test_default_pp_doclayout_v3_config() {
        let config = PPDocLayoutV3Config::default();
        assert_eq!(config.encoder_hidden_dim, 256);
        assert_eq!(config.encoder_in_channels, vec![512, 1024, 2048]);
        assert_eq!(config.feat_strides, vec![8, 16, 32]);
        assert_eq!(config.encoder_layers, 1);
        assert_eq!(config.encoder_ffn_dim, 1024);
        assert_eq!(config.encoder_attention_heads, 8);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.num_queries, 300);
        assert_eq!(config.decoder_layers, 6);
        assert_eq!(config.decoder_attention_heads, 8);
        assert_eq!(config.decoder_ffn_dim, 1024);
        assert_eq!(config.decoder_n_points, 4);
        assert_eq!(config.num_feature_levels, 3);
        assert_eq!(config.num_labels, 25);
        assert_eq!(config.num_prototypes, 32);
        assert_eq!(config.global_pointer_head_size, 64);
        assert_eq!(config.layer_norm_eps, 1e-5);
        assert_eq!(config.batch_norm_eps, 1e-5);
        assert!(config.freeze_backbone_batch_norms);
        assert!(config.mask_enhanced);
        assert_eq!(config.activation_function, "silu");
        assert_eq!(config.encoder_activation_function, "gelu");
        assert_eq!(config.decoder_activation_function, "relu");
    }

    #[test]
    fn test_id2label_mapping() {
        let config = PPDocLayoutV3Config::default();
        assert_eq!(config.id2label.len(), 25);
        assert_eq!(config.id2label.get("0"), Some(&"abstract".to_string()));
        assert_eq!(
            config.id2label.get("17"),
            Some(&"paragraph_title".to_string())
        );
        assert_eq!(config.id2label.get("21"), Some(&"table".to_string()));
        assert_eq!(config.id2label.get("22"), Some(&"text".to_string()));
        assert_eq!(
            config.id2label.get("24"),
            Some(&"vision_footnote".to_string())
        );
    }

    #[test]
    fn test_backbone_output_channels() {
        let config = PPDocLayoutV3Config::default();
        let channels = config.backbone_output_channels();
        assert_eq!(channels, vec![128, 512, 1024, 2048]);
    }

    #[test]
    fn test_deserialize_hgnetv2_config() {
        let json = r#"{
            "model_type": "hgnet_v2",
            "num_channels": 3,
            "stem_channels": [3, 32, 48],
            "stage_in_channels": [48, 128, 512, 1024],
            "stage_mid_channels": [48, 96, 192, 384],
            "stage_out_channels": [128, 512, 1024, 2048],
            "stage_num_blocks": [1, 1, 3, 1],
            "stage_downsample": [false, true, true, true],
            "stage_light_block": [false, false, true, true],
            "stage_kernel_size": [3, 3, 5, 5],
            "stage_numb_of_layers": [6, 6, 6, 6],
            "out_features": ["stage1", "stage2", "stage3", "stage4"]
        }"#;
        let config: HGNetV2Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "hgnet_v2");
        assert_eq!(config.stem_channels, vec![3, 32, 48]);
        assert_eq!(config.stage_out_channels, vec![128, 512, 1024, 2048]);
    }

    #[test]
    fn test_deserialize_full_config() {
        let json = r#"{
            "encoder_hidden_dim": 256,
            "d_model": 256,
            "num_queries": 300,
            "num_labels": 25,
            "backbone_config": {
                "model_type": "hgnet_v2",
                "stem_channels": [3, 32, 48],
                "stage_out_channels": [128, 512, 1024, 2048]
            }
        }"#;
        let config: PPDocLayoutV3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.encoder_hidden_dim, 256);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.num_queries, 300);
        assert_eq!(config.num_labels, 25);
        assert_eq!(
            config.backbone_config.stage_out_channels,
            vec![128, 512, 1024, 2048]
        );
    }

    #[test]
    fn test_fixup_derives_num_labels_from_id2label() {
        // Simulate the actual PP-DocLayoutV3 config.json which has id2label but no num_labels.
        let json = r#"{
            "id2label": {
                "0": "abstract",
                "1": "algorithm",
                "2": "aside_text"
            }
        }"#;
        let mut config: PPDocLayoutV3Config = serde_json::from_str(json).unwrap();
        // Before fixup, num_labels is the default (25) but id2label has only 3 entries
        assert_eq!(config.num_labels, 25);
        assert_eq!(config.id2label.len(), 3);

        config.fixup_after_load();
        // After fixup, num_labels should match id2label length
        assert_eq!(config.num_labels, 3);
    }

    #[test]
    fn test_feature_strides_alias() {
        // The actual model config.json uses "feature_strides" instead of "feat_strides"
        let json = r#"{
            "feature_strides": [8, 16, 32]
        }"#;
        let config: PPDocLayoutV3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.feat_strides, vec![8, 16, 32]);
    }
}
