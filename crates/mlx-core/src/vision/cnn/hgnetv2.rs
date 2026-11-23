//! Shared HGNetV2 Backbone (NHWC)
//!
//! Used by `pp_doclayout_v3`, `pp_text_det`, and `pp_text_rec`.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use serde::Deserialize;

use super::layers::{ConvBNAct, ConvBNActLight, MaxPool2d};

// ============================================================================
// HGNetV2 Backbone Configuration
// ============================================================================

/// Configuration for the HGNetV2 backbone.
///
/// Default values correspond to the HGNetV2-L / B4 architecture.
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
    #[serde(default = "default_stage_strides")]
    pub stage_strides: Vec<(i32, i32)>,

    /// Stride for stem3 layer.
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
    pub fn output_channels(&self) -> Vec<i32> {
        self.out_features
            .iter()
            .filter_map(|name| {
                if let Some(idx_str) = name.strip_prefix("stage")
                    && let Ok(idx) = idx_str.parse::<usize>()
                {
                    return self.stage_out_channels.get(idx.saturating_sub(1)).copied();
                }
                None
            })
            .collect()
    }
}

fn default_hgnet_v2_model_type() -> String {
    "hgnet_v2".to_string()
}
fn default_num_channels() -> i32 {
    3
}
fn default_embedding_size() -> i32 {
    48
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
// HGBlock: Basic building block with feature aggregation
// ============================================================================

/// A single convolutional layer within an HGBlock (standard or light).
pub enum HGBlockLayer {
    Standard(ConvBNAct),
    Light(ConvBNActLight),
}

impl HGBlockLayer {
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        match self {
            HGBlockLayer::Standard(layer) => layer.forward(input),
            HGBlockLayer::Light(layer) => layer.forward(input),
        }
    }
}

/// HGBlock: Feature aggregation block.
pub struct HGBlock {
    layers: Vec<HGBlockLayer>,
    agg_squeeze: ConvBNAct,
    agg_excitation: ConvBNAct,
    residual: bool,
}

impl HGBlock {
    pub fn new(
        layers: Vec<HGBlockLayer>,
        agg_squeeze: ConvBNAct,
        agg_excitation: ConvBNAct,
        residual: bool,
    ) -> Self {
        Self {
            layers,
            agg_squeeze,
            agg_excitation,
            residual,
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let identity = input.clone();
        let mut outputs: Vec<MxArray> = vec![input.clone()];

        let mut hidden = input.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            outputs.push(hidden.clone());
        }

        let output_refs: Vec<&MxArray> = outputs.iter().collect();
        let concatenated = MxArray::concatenate_many(output_refs, Some(3))?;

        let mut aggregated = self.agg_squeeze.forward(&concatenated)?;
        aggregated = self.agg_excitation.forward(&aggregated)?;

        if self.residual {
            aggregated = aggregated.add(&identity)?;
        }

        Ok(aggregated)
    }
}

// ============================================================================
// HGNetV2Embeddings (Stem)
// ============================================================================

/// HGNetV2 Stem (Embeddings) layer.
pub struct HGNetV2Embeddings {
    stem1: ConvBNAct,
    stem2a: ConvBNAct,
    stem2b: ConvBNAct,
    stem3: ConvBNAct,
    stem4: ConvBNAct,
    pool: MaxPool2d,
}

impl HGNetV2Embeddings {
    pub fn new(
        stem1: ConvBNAct,
        stem2a: ConvBNAct,
        stem2b: ConvBNAct,
        stem3: ConvBNAct,
        stem4: ConvBNAct,
    ) -> Self {
        Self {
            stem1,
            stem2a,
            stem2b,
            stem3,
            stem4,
            pool: MaxPool2d::new(2, 1, true),
        }
    }

    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        let embedding = self.stem1.forward(pixel_values)?;
        let padded = embedding.pad(&[0, 0, 0, 1, 0, 1, 0, 0], 0.0)?;
        let emb_stem_2a = self.stem2a.forward(&padded)?;
        let emb_stem_2a_padded = emb_stem_2a.pad(&[0, 0, 0, 1, 0, 1, 0, 0], 0.0)?;
        let emb_stem_2b = self.stem2b.forward(&emb_stem_2a_padded)?;
        let pooled_emb = self.pool.forward(&padded)?;
        let concatenated = MxArray::concatenate(&pooled_emb, &emb_stem_2b, 3)?;
        let embedding = self.stem3.forward(&concatenated)?;
        self.stem4.forward(&embedding)
    }
}

// ============================================================================
// HGNetV2Stage / HGNetV2Encoder / HGNetV2Backbone
// ============================================================================

/// HGNetV2 Stage: optional depthwise downsample + stack of HGBlocks.
pub struct HGNetV2Stage {
    downsample: Option<ConvBNAct>,
    blocks: Vec<HGBlock>,
}

impl HGNetV2Stage {
    pub fn new(downsample: Option<ConvBNAct>, blocks: Vec<HGBlock>) -> Self {
        Self { downsample, blocks }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = if let Some(ref ds) = self.downsample {
            ds.forward(input)?
        } else {
            input.clone()
        };

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        Ok(x)
    }
}

/// HGNetV2 Encoder: sequence of stages.
pub struct HGNetV2Encoder {
    stages: Vec<HGNetV2Stage>,
}

impl HGNetV2Encoder {
    pub fn new(stages: Vec<HGNetV2Stage>) -> Self {
        Self { stages }
    }

    pub fn forward(&self, input: &MxArray) -> Result<Vec<MxArray>> {
        let mut hidden_states: Vec<MxArray> = Vec::with_capacity(self.stages.len() + 1);
        hidden_states.push(input.clone());

        let mut x = input.clone();
        for stage in &self.stages {
            x = stage.forward(&x)?;
            hidden_states.push(x.clone());
        }

        Ok(hidden_states)
    }
}

/// HGNetV2 Backbone: Embeddings + Encoder + feature selection.
pub struct HGNetV2Backbone {
    embedder: HGNetV2Embeddings,
    encoder: HGNetV2Encoder,
    out_feature_indices: Vec<usize>,
    pub channels: Vec<i32>,
}

impl HGNetV2Backbone {
    pub fn new(
        embedder: HGNetV2Embeddings,
        encoder: HGNetV2Encoder,
        config: &HGNetV2Config,
    ) -> Self {
        let stage_names: Vec<String> = std::iter::once("stem".to_string())
            .chain((1..=config.num_stages()).map(|i| format!("stage{i}")))
            .collect();

        let out_feature_indices: Vec<usize> = config
            .out_features
            .iter()
            .filter_map(|name| stage_names.iter().position(|s| s == name))
            .collect();

        let all_channels: Vec<i32> = std::iter::once(config.embedding_size)
            .chain(config.stage_out_channels.iter().copied())
            .collect();

        let channels: Vec<i32> = out_feature_indices
            .iter()
            .map(|&idx| all_channels[idx])
            .collect();

        Self {
            embedder,
            encoder,
            out_feature_indices,
            channels,
        }
    }

    pub fn forward(&self, pixel_values: &MxArray) -> Result<Vec<MxArray>> {
        let embedding_output = self.embedder.forward(pixel_values)?;
        let hidden_states = self.encoder.forward(&embedding_output)?;

        let feature_maps: Vec<MxArray> = self
            .out_feature_indices
            .iter()
            .map(|&idx| hidden_states[idx].clone())
            .collect();

        Ok(feature_maps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HGNetV2Config::default();
        assert_eq!(config.num_stages(), 4);
        assert_eq!(config.stem_channels, vec![3, 32, 48]);
        assert_eq!(config.output_channels(), vec![128, 512, 1024, 2048]);
    }
}
