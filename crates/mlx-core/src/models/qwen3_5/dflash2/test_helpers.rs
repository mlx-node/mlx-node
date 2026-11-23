use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;

use super::DFlash2Model;
use super::compiled::CompiledDraft;
use super::config::{DFlash2Config, DraftExecutionPolicy};
use super::layers::{DFlash2Attention, DFlash2Layer, DFlash2Mlp, GroupedDynamicCausalConv};
use super::loader::{expected_tensor_shapes, read_dflash2_config};
use super::selector::CandidateSelector;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm, RoPE};

pub(crate) fn tiny_dflash2_model_for_stepper_test(
    target: &super::super::config::Qwen3_5Config,
) -> Result<DFlash2Model> {
    tiny_dflash2_model_with_block(target, 2)
}

/// Writes a DFlash2 checkpoint of `config` (a `config.json` value) to
/// `dir`: random BF16 tensors of every expected shape.
pub(crate) fn write_test_checkpoint(dir: &Path, config: &serde_json::Value) -> Result<()> {
    fs::create_dir_all(dir).map_err(|error| Error::from_reason(error.to_string()))?;
    fs::write(dir.join("config.json"), config.to_string())
        .map_err(|error| Error::from_reason(error.to_string()))?;
    let parsed = read_dflash2_config(dir)?;
    let mut tensors = HashMap::new();
    for (name, shape) in expected_tensor_shapes(&parsed) {
        let array = MxArray::random_normal(&shape, 0.0, 0.02, Some(DType::BFloat16))?;
        tensors.insert(name, array);
    }
    crate::utils::safetensors::save_safetensors(dir.join("model.safetensors"), &mut tensors, None)
}

/// A one-layer draft geometry for window-only tests.
pub(crate) fn tiny_dflash2_config() -> DFlash2Config {
    DFlash2Config {
        block_size: 2,
        mask_token_id: 0,
        target_layers: vec![0],
        target_num_layers: 1,
        hidden_size: 64,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 8,
        vocab_size: 16,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 64,
        sliding_window: 8,
        conv_group_size: 8,
        conv_kernel_size: 2,
        selector_rank: 8,
        selector_top_k: 4,
        rope_theta: 1e4,
    }
}

/// The tiny draft geometry with a context row `hidden` wide and `window`
/// sliding rows.
pub(crate) fn dflash2_config_with_context(hidden: usize, window: usize) -> DFlash2Config {
    DFlash2Config {
        hidden_size: hidden,
        sliding_window: window,
        ..tiny_dflash2_config()
    }
}

pub(crate) fn tiny_dflash2_model_with_block(
    target: &super::super::config::Qwen3_5Config,
    block_size: usize,
) -> Result<DFlash2Model> {
    let execution_policy = DraftExecutionPolicy::capture();
    let hidden = target.hidden_size as usize;
    let head_dim = target.head_dim as usize;
    let config = DFlash2Config {
        block_size,
        mask_token_id: 0,
        target_layers: vec![1, 3],
        target_num_layers: target.num_layers as usize,
        hidden_size: hidden,
        intermediate_size: hidden * 2,
        num_hidden_layers: 1,
        num_attention_heads: target.num_heads as usize,
        num_key_value_heads: target.num_kv_heads as usize,
        head_dim,
        vocab_size: target.vocab_size as usize,
        rms_norm_eps: target.rms_norm_eps,
        max_position_embeddings: target.max_position_embeddings as usize,
        sliding_window: 8,
        conv_group_size: head_dim,
        conv_kernel_size: 2,
        selector_rank: 8,
        selector_top_k: 4,
        rope_theta: target.rope_theta,
    };
    if config
        .target_layers
        .iter()
        .any(|&layer| layer >= config.target_num_layers)
        || !hidden.is_multiple_of(config.conv_group_size)
        || config.vocab_size < config.selector_top_k
    {
        return Err(Error::from_reason(
            "tiny DFlash2 stepper fixture is incompatible with the target config",
        ));
    }

    let linear = |input: usize, output: usize| -> Result<LinearProj> {
        let mut layer = Linear::new(input as u32, output as u32, Some(false))?;
        layer.set_weight(&MxArray::random_normal(
            &[output as i64, input as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )?)?;
        Ok(LinearProj::Standard(layer))
    };
    let norm = |size: usize| -> Result<RMSNorm> {
        let mut layer = RMSNorm::new(size as u32, Some(config.rms_norm_eps))?;
        layer.set_weight(&MxArray::ones(&[size as i64], Some(DType::BFloat16))?)?;
        Ok(layer)
    };
    let embedding = |rows: usize, width: usize| -> Result<Embedding> {
        let mut layer = Embedding::new(rows as u32, width as u32)?;
        layer.load_weight(&MxArray::random_normal(
            &[rows as i64, width as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )?)?;
        Ok(layer)
    };
    let grouped_conv = || -> Result<GroupedDynamicCausalConv> {
        Ok(GroupedDynamicCausalConv {
            source_enabled: false,
            base_kernel: MxArray::random_normal(
                &[2, config.conv_kernel_size as i64, hidden as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )?,
            kernel_projection: linear(
                hidden,
                2 * config.conv_kernel_size * (hidden / config.conv_group_size),
            )?,
            kernel_size: config.conv_kernel_size,
            group_size: config.conv_group_size,
        })
    };
    let attention = DFlash2Attention {
        tensor_attention_enabled: execution_policy.tensor_attention(),
        q_proj: linear(hidden, config.num_attention_heads * head_dim)?,
        k_proj: linear(hidden, config.num_key_value_heads * head_dim)?,
        v_proj: linear(hidden, config.num_key_value_heads * head_dim)?,
        o_proj: linear(config.num_attention_heads * head_dim, hidden)?,
        qkv_proj: None,
        q_norm: norm(head_dim)?,
        k_norm: norm(head_dim)?,
        rope: RoPE::new(head_dim as i32, Some(false), Some(config.rope_theta), None),
        num_heads: config.num_attention_heads as i64,
        num_kv_heads: config.num_key_value_heads as i64,
        head_dim: head_dim as i64,
    };
    let layer = DFlash2Layer {
        attention,
        mlp: DFlash2Mlp {
            gate_proj: linear(hidden, config.intermediate_size)?,
            up_proj: linear(hidden, config.intermediate_size)?,
            down_proj: linear(config.intermediate_size, hidden)?,
            gate_up: None,
        },
        input_norm: norm(hidden)?,
        post_attention_norm: norm(hidden)?,
        attention_conv: grouped_conv()?,
        mlp_conv: grouped_conv()?,
    };
    Ok(DFlash2Model {
        execution_policy,
        fc: linear(hidden * config.target_layers.len(), hidden)?,
        hidden_norm: norm(hidden)?,
        norm: norm(hidden)?,
        selector: CandidateSelector {
            predecessor_codebook: embedding(config.vocab_size, config.selector_rank)?,
            successor_codebook: embedding(config.vocab_size, config.selector_rank)?,
            hidden_projection: linear(hidden, config.selector_rank)?,
            top_k: config.selector_top_k,
            rank: config.selector_rank,
            vocab_size: config.vocab_size,
        },
        layers: vec![layer],
        config,
        weight_bytes: 0,
        context_kv: None,
        compiled: CompiledDraft::default(),
    })
}
