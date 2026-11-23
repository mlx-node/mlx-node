use napi::bindgen_prelude::*;
use serde::Deserialize;

use super::{DFLASH2_DRAFT_PROPOSALS, DFLASH2_VERIFY_ROWS, DFlash2TargetShape};
use crate::models::qwen3_5::draft_context::{DRAFT_CONTEXT_GROUP, DraftContextWindow};

#[derive(Clone, Debug, Deserialize)]
struct RawDFlashConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct RawConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_target_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    sliding_window: usize,
    layer_types: Vec<String>,
    #[serde(default)]
    is_causal: bool,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    dflash_config: RawDFlashConfig,
}

#[derive(Clone, Debug)]
pub(crate) struct DFlash2Config {
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: usize,
    pub(crate) target_layers: Vec<usize>,
    pub(super) target_num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) num_hidden_layers: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    pub(super) vocab_size: usize,
    pub(super) rms_norm_eps: f64,
    pub(super) max_position_embeddings: usize,
    pub(crate) sliding_window: usize,
    pub(super) conv_group_size: usize,
    pub(super) conv_kernel_size: usize,
    pub(super) selector_rank: usize,
    pub(super) selector_top_k: usize,
    pub(super) rope_theta: f64,
}

impl DFlash2Config {
    pub(crate) fn validate_target(&self, target: DFlash2TargetShape) -> Result<()> {
        let invalid_tap = self
            .target_layers
            .iter()
            .copied()
            .find(|&layer| layer >= target.num_layers);
        if self.hidden_size != target.hidden_size
            || self.vocab_size != target.vocab_size
            || self.mask_token_id >= self.vocab_size
            || self.target_num_layers != target.num_layers
            || invalid_tap.is_some()
        {
            return Err(Error::from_reason(format!(
                "DFlash2/target mismatch: draft hidden={} vocab={} taps={:?}; target hidden={} vocab={} layers={}",
                self.hidden_size,
                self.vocab_size,
                self.target_layers,
                target.hidden_size,
                target.vocab_size,
                target.num_layers,
            )));
        }
        Ok(())
    }

    pub(super) fn from_raw(raw: RawConfig) -> Result<Self> {
        if !raw
            .architectures
            .iter()
            .any(|name| name == "DFlash2DraftModel")
        {
            return Err(Error::from_reason(
                "DFlash2 checkpoint architectures must contain DFlash2DraftModel",
            ));
        }
        let draft = raw.dflash_config;
        let rope_theta = raw
            .rope_theta
            .or_else(|| {
                raw.rope_parameters
                    .as_ref()
                    .and_then(|value| value.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
            })
            .unwrap_or(10_000.0);
        let valid = raw.hidden_size > 0
            && raw.intermediate_size > 0
            && raw.num_hidden_layers > 0
            && raw.num_attention_heads > 0
            && raw.num_key_value_heads > 0
            && raw
                .num_attention_heads
                .is_multiple_of(raw.num_key_value_heads)
            && raw.head_dim > 0
            && raw.vocab_size > 0
            && raw.sliding_window > 1
            && raw.layer_types.len() == raw.num_hidden_layers
            && raw
                .layer_types
                .iter()
                .all(|kind| kind == "sliding_attention")
            && !raw.is_causal
            && draft.conv_kernel_size == 2
            && draft.conv_group_size > 0
            && raw.hidden_size.is_multiple_of(draft.conv_group_size)
            && draft.selector_rank > 0
            && draft.selector_top_k > 0
            && draft.selector_top_k <= raw.vocab_size
            && !draft.target_layer_ids.is_empty()
            && raw.num_target_layers > 0
            && draft
                .target_layer_ids
                .iter()
                .all(|&layer| layer < raw.num_target_layers)
            && draft
                .target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]);
        if !valid {
            return Err(Error::from_reason(
                "unsupported DFlash2 configuration (requires non-causal sliding attention, two-tap grouped convolution, and a non-empty selector)",
            ));
        }
        // Splash `DraftAttention::contextWindowBytes`: the 4-bit context
        // window quantizes whole 64-value groups of a context row.
        if !raw.hidden_size.is_multiple_of(DRAFT_CONTEXT_GROUP as usize) {
            return Err(Error::from_reason(format!(
                "draft config hidden_size {} is not a multiple of the {DRAFT_CONTEXT_GROUP}-value context group",
                raw.hidden_size
            )));
        }
        // Splash `ModelDescriptor` draft check: the trained block holds the
        // verify rows or more, and a cycle drafts the first of them.
        if draft.block_size < DFLASH2_VERIFY_ROWS {
            return Err(Error::from_reason(format!(
                "draft config dflash_config block_size {} is less than the {DFLASH2_VERIFY_ROWS} rows a cycle drafts",
                draft.block_size
            )));
        }
        Ok(Self {
            block_size: DFLASH2_DRAFT_PROPOSALS,
            mask_token_id: draft.mask_token_id,
            target_layers: draft.target_layer_ids,
            target_num_layers: raw.num_target_layers,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            conv_group_size: draft.conv_group_size,
            conv_kernel_size: draft.conv_kernel_size,
            selector_rank: draft.selector_rank,
            selector_top_k: draft.selector_top_k,
            rope_theta,
        })
    }
}

impl DFlash2Config {
    /// Bytes of every layer's BF16 K and V draft ring: `sliding_window`
    /// slots and the rows a proposal's block lands in past them.
    pub(crate) fn kv_ring_bytes(&self) -> u64 {
        let rows = (self.sliding_window as u64)
            .max(2)
            .saturating_add(self.block_size as u64 + 1);
        (self.num_hidden_layers as u64)
            .saturating_mul(2)
            .saturating_mul(self.num_key_value_heads as u64)
            .saturating_mul(self.head_dim as u64)
            .saturating_mul(2)
            .saturating_mul(rows)
    }

    /// A lane's draft buffers beyond its GDN cells: the K/V ring and the
    /// 4-bit context window (Splash `CompositeStateLayout::laneBytes`
    /// without the cells).
    pub(crate) fn lane_bytes(&self) -> u64 {
        self.kv_ring_bytes()
            .saturating_add(self.context_window_bytes())
    }

    /// Bytes of the 4-bit context window a state holds (Splash
    /// `DraftStateLayout::windowBytes`).
    pub(crate) fn context_window_bytes(&self) -> u64 {
        DraftContextWindow::bytes(self.hidden_size as u32, self.sliding_window as u32)
    }

    /// The namespace's `draft` line: layers, KV heads, head size and sliding
    /// window (Splash `persistentCacheNamespace`).
    pub(crate) fn namespace_geometry(&self) -> (u32, u32, u32, u32) {
        (
            self.num_hidden_layers as u32,
            self.num_key_value_heads as u32,
            self.head_dim as u32,
            self.sliding_window as u32,
        )
    }

    /// The context window's row width (the namespace's `context` line).
    pub(crate) fn context_width(&self) -> u32 {
        self.hidden_size as u32
    }

    /// An empty context window of this draft's geometry.
    pub(crate) fn context_window(&self) -> Result<DraftContextWindow> {
        DraftContextWindow::new(self.hidden_size as u32, self.sliding_window as u32)
    }

    pub(super) fn supports_source_conv(&self, target: [i32; 5]) -> bool {
        self.supports_moe_execution_defaults(target)
            && self.target_layers == [1, 6, 11, 16, 22, 27, 32, 37]
            && self.mask_token_id == 248077
            && self.vocab_size == 248320
            && self.max_position_embeddings == 262144
            && (self.rms_norm_eps as f32).to_bits() == 1e-6f32.to_bits()
            && (self.rope_theta as f32).to_bits() == 10000000.0f32.to_bits()
    }

    pub(super) fn supports_moe_execution_defaults(&self, target: [i32; 5]) -> bool {
        target == [2048, 40, 256, 8, 512]
            && self.hidden_size == 2048
            && self.intermediate_size == 6144
            && self.target_num_layers == 40
            && self.target_layers.len() == 8
            && self.num_hidden_layers == 6
            && self.num_attention_heads == 32
            && self.num_key_value_heads == 8
            && self.head_dim == 128
            && self.sliding_window == 2048
            && self.block_size == 7
            && self.conv_group_size == 16
            && self.conv_kernel_size == 2
            && self.selector_rank == 256
            && self.selector_top_k == 16
    }
}

/// Captured once per loaded companion. A MoE loader may select its measured
/// structural profile before publishing the model; later environment changes
/// cannot alter a cached graph or another model's execution policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DraftExecutionPolicy {
    pub(super) compile_override: Option<bool>,
    attention_override: Option<bool>,
    supported_moe: bool,
    dense_cuda: bool,
    source_conv_requested: bool,
    source_conv_supported: bool,
}

impl DraftExecutionPolicy {
    #[cfg(test)]
    pub(super) fn from_values(
        compile: Option<&str>,
        attention: Option<&str>,
        source_conv: Option<&str>,
    ) -> Self {
        Self {
            compile_override: compile.map(|value| value == "1"),
            attention_override: attention.map(|value| value == "1"),
            supported_moe: false,
            dense_cuda: false,
            source_conv_requested: source_conv == Some("1"),
            source_conv_supported: false,
        }
    }

    pub(super) fn capture() -> Self {
        Self {
            compile_override: std::env::var_os("MLX_DFLASH2_COMPILE_DRAFT")
                .map(|value| value == "1"),
            attention_override: std::env::var_os("MLX_DFLASH2_ATTENTION_NAX")
                .map(|value| value == "1"),
            supported_moe: false,
            dense_cuda: false,
            source_conv_requested: std::env::var_os("MLX_DFLASH2_SOURCE_CONV")
                .is_some_and(|value| value == "1"),
            source_conv_supported: false,
        }
    }

    pub(super) fn with_moe_profile(mut self, supported: bool) -> Self {
        self.supported_moe = supported;
        self
    }

    pub(super) fn with_source_conv_profile(mut self, supported: bool) -> Self {
        self.source_conv_supported = supported;
        self
    }

    pub(super) fn source_conv(self) -> bool {
        self.source_conv_requested && self.supported_moe && self.source_conv_supported
    }

    pub(super) fn with_dense_cuda_profile(mut self, supported: bool) -> Self {
        self.dense_cuda = supported;
        self
    }

    pub(super) fn compile_draft(self) -> bool {
        self.compile_override
            .unwrap_or(self.supported_moe || self.dense_cuda)
    }

    pub(super) fn tensor_attention(self) -> bool {
        self.attention_override.unwrap_or(self.supported_moe)
    }
}
