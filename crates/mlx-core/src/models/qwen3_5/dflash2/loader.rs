use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::DFlash2Model;
use super::compiled::CompiledDraft;
use super::config::{DFlash2Config, DraftExecutionPolicy, RawConfig};
use super::layers::{DFlash2Attention, DFlash2Layer, DFlash2Mlp, GroupedDynamicCausalConv};
use super::selector::CandidateSelector;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
use crate::nn::{Embedding, Linear, RMSNorm, RoPE};
use crate::utils::safetensors::load_safetensors_lazy;

fn required(params: &mut HashMap<String, MxArray>, key: &str, shape: &[i64]) -> Result<MxArray> {
    let value = params
        .remove(key)
        .ok_or_else(|| Error::from_reason(format!("DFlash2 checkpoint is missing '{key}'")))?;
    validate_tensor(&value, key, shape)?;
    Ok(value)
}

fn validate_tensor(value: &MxArray, key: &str, shape: &[i64]) -> Result<()> {
    if value.shape()?.as_ref() != shape {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' has shape {:?}, expected {shape:?}",
            value.shape()?.as_ref()
        )));
    }
    let actual = value.dtype()?;
    if !matches!(actual, DType::Float16 | DType::BFloat16 | DType::Float32) {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' has dtype {actual:?}, expected floating point"
        )));
    }
    Ok(())
}

/// The published DFlash2 companion ships bf16. Its dense projections load as
/// affine Q4/group64: against bf16 and Q8 it kept teacher-forced acceptance
/// within noise and gave the lowest decode time per committed token. On a
/// Metal host the packed arrays are then tiled into the K-quant `a4g64@t64`
/// contract (same codes, same bf16 scale and bias per group), so a decode
/// block's 8 rows take the tensor-op `qmm_m8_nax_t64` kernel instead of
/// MLX's per-row affine `qmv` (which read the weights once per row: ~2.2x
/// the bandwidth floor at M = 8). The draft reuses the target output head.
/// A change of draft precision moves proposals and verify grouping, so it
/// can change the transcript even though the target verifies every emitted
/// token.
const DRAFT_GROUP_SIZE: i32 = 64;
const DRAFT_BITS: i32 = 4;

/// Affine-quantize a floating `[out, in]` weight, returning MLX's packed
/// `(weight, scales, biases)` triple consumed by `quantized_matmul`.
fn quantize_affine(
    weight: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let mut out_b = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_quantize(
            weight.as_raw_ptr(),
            group_size,
            bits,
            c"affine".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_quantize(affine) failed on a DFlash2 draft weight",
        ));
    }
    let packed = MxArray::from_handle(out_q, "dflash2 quantized weight")?;
    let scales = MxArray::from_handle(out_s, "dflash2 quantization scales")?;
    let biases = MxArray::from_handle(out_b, "dflash2 quantization biases")?;
    // mlx_quantize outputs are lazy: left unevaluated they pin the dense bf16
    // source in the graph until the first draft forward, so residency sizing
    // would run against ~3.6 GB of still-live dense weights and the first
    // proposal would pay the whole quantize cost. Evaluating here releases
    // each dense source when the caller drops it.
    packed.eval();
    scales.eval();
    biases.eval();
    Ok((packed, scales, biases))
}

/// Quantize a floating `[out, in]` weight to affine Q4/group64 as a
/// `QuantizedLinear`, tiled into the `a4g64@t64` K-quant contract when this
/// host runs the `_t64` kernels ([`QuantizedLinear::tile_kquant_layout_any_shape`]:
/// the draft's widths were gated on their own numbers, not the generic
/// affine shape rule; the row-major arrays are released). The shape must be tileable
/// (`N % 64 == 0`, `K % 256 == 0`) on every host: an odd draft geometry
/// fails here rather than silently taking a slower layout.
pub(super) fn quantize_draft_weight(weight: &MxArray, name: &str) -> Result<QuantizedLinear> {
    use crate::models::quant_dispatch::{kquant_tileable, kquant_tiled_enabled};
    let shape = weight.shape()?;
    let (rows, k) = (shape[0], shape[1]);
    if !kquant_tileable(rows, k) {
        return Err(Error::from_reason(format!(
            "DFlash2 draft projection '{name}' is [{rows}, {k}]; the a{DRAFT_BITS}g{DRAFT_GROUP_SIZE} \
             Tiled64 layout needs N % 64 == 0 and K % 256 == 0"
        )));
    }
    let (packed, scales, biases) = quantize_affine(weight, DRAFT_GROUP_SIZE, DRAFT_BITS)?;
    let mut linear = QuantizedLinear::new(
        packed,
        scales,
        Some(biases),
        None,
        DRAFT_GROUP_SIZE,
        DRAFT_BITS,
        "affine".to_string(),
    );
    if kquant_tiled_enabled() && !linear.tile_kquant_layout_any_shape()? {
        return Err(Error::from_reason(format!(
            "DFlash2 draft projection '{name}' [{rows}, {k}] did not tile"
        )));
    }
    Ok(linear)
}

/// Builds a draft projection as affine Q4/group64. `savings` accumulates
/// `dense − resident` bytes so the loader can report true residency rather
/// than the bf16 file size.
pub(super) fn draft_linear(
    params: &mut HashMap<String, MxArray>,
    prefix: &str,
    input: usize,
    output: usize,
    savings: &mut u64,
) -> Result<LinearProj> {
    let key = format!("{prefix}.weight");
    let weight = required(params, &key, &[output as i64, input as i64])?;
    let linear = quantize_draft_weight(&weight, &key)?;
    let resident = linear.get_weight().nbytes() as u64
        + linear.get_scales().nbytes() as u64
        + linear.get_biases().map_or(0, |b| b.nbytes() as u64);
    *savings += (weight.nbytes() as u64).saturating_sub(resident);
    Ok(LinearProj::Quantized(linear))
}

pub(super) fn dense_linear(
    params: &mut HashMap<String, MxArray>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<LinearProj> {
    let weight = required(
        params,
        &format!("{prefix}.weight"),
        &[output as i64, input as i64],
    )?;
    Ok(LinearProj::Standard(Linear::from_weights(&weight, None)?))
}

fn norm(
    params: &mut HashMap<String, MxArray>,
    key: &str,
    size: usize,
    eps: f64,
) -> Result<RMSNorm> {
    RMSNorm::from_weight(&required(params, key, &[size as i64])?, Some(eps))
}

fn codebook(
    params: &mut HashMap<String, MxArray>,
    key: &str,
    vocab: usize,
    rank: usize,
) -> Result<Embedding> {
    let weight = required(params, key, &[vocab as i64, rank as i64])?;
    let mut embedding = Embedding::new_uninitialized(vocab as u32, rank as u32)?;
    embedding.load_weight(&weight)?;
    Ok(embedding)
}

fn grouped_conv(
    params: &mut HashMap<String, MxArray>,
    base: &str,
    name: &str,
    config: &DFlash2Config,
    savings: &mut u64,
) -> Result<GroupedDynamicCausalConv> {
    let hidden = config.hidden_size;
    let groups = hidden / config.conv_group_size;
    Ok(GroupedDynamicCausalConv {
        source_enabled: false,
        base_kernel: required(
            params,
            &format!("{base}.{name}.base_kernel"),
            &[2, config.conv_kernel_size as i64, hidden as i64],
        )?,
        kernel_projection: draft_linear(
            params,
            &format!("{base}.{name}.kernel_projection"),
            hidden,
            2 * config.conv_kernel_size * groups,
            savings,
        )?,
        kernel_size: config.conv_kernel_size,
        group_size: config.conv_group_size,
    })
}

pub(super) fn resolve_safetensors(path: &Path) -> Result<PathBuf> {
    if path.join("model.safetensors").is_file() {
        return Ok(path.join("model.safetensors"));
    }
    Err(Error::from_reason(format!(
        "DFlash2 checkpoint {} is missing model.safetensors",
        path.display()
    )))
}

pub(super) fn expected_tensor_shapes(config: &DFlash2Config) -> Vec<(String, Vec<i64>)> {
    let hidden = config.hidden_size as i64;
    let intermediate = config.intermediate_size as i64;
    let groups = config.hidden_size / config.conv_group_size;
    let mut expected = vec![
        (
            "fc.weight".to_string(),
            vec![hidden, hidden * config.target_layers.len() as i64],
        ),
        ("hidden_norm.weight".to_string(), vec![hidden]),
        ("norm.weight".to_string(), vec![hidden]),
    ];
    for index in 0..config.num_hidden_layers {
        let base = format!("layers.{index}");
        let attention = format!("{base}.self_attn");
        expected.extend([
            (
                format!("{attention}.q_proj.weight"),
                vec![
                    (config.num_attention_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.k_proj.weight"),
                vec![
                    (config.num_key_value_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.v_proj.weight"),
                vec![
                    (config.num_key_value_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.o_proj.weight"),
                vec![
                    hidden,
                    (config.num_attention_heads * config.head_dim) as i64,
                ],
            ),
            (
                format!("{attention}.q_norm.weight"),
                vec![config.head_dim as i64],
            ),
            (
                format!("{attention}.k_norm.weight"),
                vec![config.head_dim as i64],
            ),
            (
                format!("{base}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{base}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{base}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ),
            (format!("{base}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{base}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{base}.attention_conv.base_kernel"),
                vec![2, config.conv_kernel_size as i64, hidden],
            ),
            (
                format!("{base}.attention_conv.kernel_projection.weight"),
                vec![(2 * config.conv_kernel_size * groups) as i64, hidden],
            ),
            (
                format!("{base}.mlp_conv.base_kernel"),
                vec![2, config.conv_kernel_size as i64, hidden],
            ),
            (
                format!("{base}.mlp_conv.kernel_projection.weight"),
                vec![(2 * config.conv_kernel_size * groups) as i64, hidden],
            ),
        ]);
    }
    expected.extend([
        (
            "candidate_selector.predecessor_codebook".to_string(),
            vec![config.vocab_size as i64, config.selector_rank as i64],
        ),
        (
            "candidate_selector.successor_codebook".to_string(),
            vec![config.vocab_size as i64, config.selector_rank as i64],
        ),
        (
            "candidate_selector.hidden_projection.weight".to_string(),
            vec![config.selector_rank as i64, hidden],
        ),
    ]);
    expected
}

pub(super) fn validate_tensor_inventory(
    params: &HashMap<String, MxArray>,
    config: &DFlash2Config,
) -> Result<()> {
    let expected = expected_tensor_shapes(config);
    let expected_names = expected
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut missing = Vec::new();
    for (name, shape) in &expected {
        let Some(array) = params.get(name) else {
            missing.push(name.clone());
            continue;
        };
        validate_tensor(array, name, shape)?;
    }
    let mut unexpected = params
        .keys()
        .filter(|name| !expected_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    unexpected.sort();
    if !missing.is_empty() || !unexpected.is_empty() {
        missing.sort();
        return Err(Error::from_reason(format!(
            "DFlash2 tensor inventory mismatch: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    Ok(())
}

/// The validated `config.json` of the DFlash2 checkpoint directory `path`.
pub(crate) fn read_dflash2_config(path: &Path) -> Result<DFlash2Config> {
    if !path.is_dir() {
        return Err(Error::from_reason(format!(
            "DFlash2 path is not a directory: {}",
            path.display()
        )));
    }
    let config_data = fs::read_to_string(path.join("config.json")).map_err(|error| {
        Error::from_reason(format!("Failed to read DFlash2 config.json: {error}"))
    })?;
    let raw: RawConfig = serde_json::from_str(&config_data).map_err(|error| {
        Error::from_reason(format!("Failed to parse DFlash2 config.json: {error}"))
    })?;
    DFlash2Config::from_raw(raw)
}

/// Bytes of the DFlash2 checkpoint file in `path`, before it loads.
pub(crate) fn dflash2_file_bytes(path: &Path) -> Result<u64> {
    let tensor_path = resolve_safetensors(path)?;
    fs::metadata(&tensor_path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            Error::from_reason(format!("Failed to stat {}: {error}", tensor_path.display()))
        })
}

pub(crate) fn load_dflash2(path: &Path) -> Result<(DFlash2Model, u64)> {
    let config = read_dflash2_config(path)?;
    let execution_policy = DraftExecutionPolicy::capture();
    let tensor_path = resolve_safetensors(path)?;
    crate::engine::persistence::prewarm_checkpoint_pages(path);
    let mut params = load_safetensors_lazy(&tensor_path)?;
    // Structural preflight precedes the first GPU evaluation. A descriptor-
    // valid but incomplete companion must fail before any target or draft
    // state can be mutated.
    validate_tensor_inventory(&params, &config)?;
    let weight_bytes = params.values().fold(0u64, |total, array| {
        total.saturating_add(array.nbytes() as u64)
    });
    let arrays = params.values().collect::<Vec<_>>();
    let _resident = crate::array::memory::materialize_weights(&arrays)?;

    let hidden = config.hidden_size;
    let mut savings = 0u64;
    let fc = draft_linear(
        &mut params,
        "fc",
        hidden * config.target_layers.len(),
        hidden,
        &mut savings,
    )?;
    let hidden_norm = norm(
        &mut params,
        "hidden_norm.weight",
        hidden,
        config.rms_norm_eps,
    )?;
    let final_norm = norm(&mut params, "norm.weight", hidden, config.rms_norm_eps)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let base = format!("layers.{index}");
        let attention_base = format!("{base}.self_attn");
        let attention = DFlash2Attention {
            tensor_attention_enabled: execution_policy.tensor_attention(),
            q_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.q_proj"),
                hidden,
                config.num_attention_heads * config.head_dim,
                &mut savings,
            )?,
            k_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.k_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                &mut savings,
            )?,
            v_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.v_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                &mut savings,
            )?,
            o_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.o_proj"),
                config.num_attention_heads * config.head_dim,
                hidden,
                &mut savings,
            )?,
            qkv_proj: None,
            q_norm: norm(
                &mut params,
                &format!("{attention_base}.q_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            k_norm: norm(
                &mut params,
                &format!("{attention_base}.k_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            rope: RoPE::new(
                config.head_dim as i32,
                Some(false),
                Some(config.rope_theta),
                None,
            ),
            num_heads: config.num_attention_heads as i64,
            num_kv_heads: config.num_key_value_heads as i64,
            head_dim: config.head_dim as i64,
        };
        let mlp_base = format!("{base}.mlp");
        let mlp = DFlash2Mlp {
            gate_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.gate_proj"),
                hidden,
                config.intermediate_size,
                &mut savings,
            )?,
            up_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.up_proj"),
                hidden,
                config.intermediate_size,
                &mut savings,
            )?,
            down_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.down_proj"),
                config.intermediate_size,
                hidden,
                &mut savings,
            )?,
            gate_up: None,
        };
        let input_norm = norm(
            &mut params,
            &format!("{base}.input_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
        )?;
        let post_attention_norm = norm(
            &mut params,
            &format!("{base}.post_attention_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
        )?;
        let attention_conv =
            grouped_conv(&mut params, &base, "attention_conv", &config, &mut savings)?;
        let mlp_conv = grouped_conv(&mut params, &base, "mlp_conv", &config, &mut savings)?;
        layers.push(DFlash2Layer {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
            attention_conv,
            mlp_conv,
        });
    }
    let selector = CandidateSelector {
        predecessor_codebook: codebook(
            &mut params,
            "candidate_selector.predecessor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?,
        successor_codebook: codebook(
            &mut params,
            "candidate_selector.successor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?,
        // The diagnostic opt-in reproduces Splash's affine Q4/group64
        // selector projection for matched-source performance/acceptance A/B.
        // Ordinary loads preserve the published BF16 selector precision.
        hidden_projection: if std::env::var("MLX_DFLASH2_SELECTOR_QUANT").is_ok_and(|v| v == "1") {
            draft_linear(
                &mut params,
                "candidate_selector.hidden_projection",
                hidden,
                config.selector_rank,
                &mut savings,
            )?
        } else {
            dense_linear(
                &mut params,
                "candidate_selector.hidden_projection",
                hidden,
                config.selector_rank,
            )?
        },
        top_k: config.selector_top_k,
        rank: config.selector_rank,
        vocab_size: config.vocab_size,
    };
    if !params.is_empty() {
        let mut unexpected = params.keys().cloned().collect::<Vec<_>>();
        unexpected.sort();
        return Err(Error::from_reason(format!(
            "DFlash2 checkpoint contains unexpected tensors: {unexpected:?}"
        )));
    }
    let weight_bytes = weight_bytes.saturating_sub(savings);
    let mut model = DFlash2Model {
        execution_policy,
        config,
        weight_bytes,
        fc,
        hidden_norm,
        layers,
        norm: final_norm,
        selector,
        context_kv: None,
        compiled: CompiledDraft::default(),
    };
    model.merge_projections()?;
    let weight_bytes = model.weight_bytes;
    Ok((model, weight_bytes))
}
