use std::collections::{HashMap, HashSet};

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quant_dispatch::{PerLayerMode, PerLayerQuant, effective_plq_for};
use crate::models::qwen3_5::Qwen3_5Config;

pub(crate) const PRISM_HADAMARD_CONFIG_KEY: &str = "prism_hadamard";
const EXPECTED_VERSION: u32 = 1;
const EXPECTED_BLOCK_SIZE: i32 = 1024;
const EXPECTED_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const EXPECTED_AXIS: &str = "input-last-dimension";
const EXPECTED_SIGN_MODE: &str = "explicit";
const EXPECTED_MODEL_TYPE: &str = "qwen3_5";
const EMBEDDING_WEIGHT_NAME: &str = "embedding.weight";
const LM_HEAD_PREFIX: &str = "lm_head";
const LAYERS_PREFIX: &str = "layers.";
const WEIGHT_SUFFIX: &str = ".weight";
const GDN_OUT_PROJ_SUFFIX: &str = "linear_attn.out_proj";
const PQ2_BITS: i32 = 2;
const PQ2_GROUP_SIZE: i32 = 128;
const PQ2_CODES_PER_WORD: i64 = 16;

const SUPPORTED_LAYER_PROJECTIONS: [&str; 10] = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PrismHadamardConfig {
    pub version: u32,
    pub block_size: i32,
    pub transform: String,
    pub axis: String,
    pub sign_mode: String,
    pub weight_names: Vec<String>,
    pub inverse_weight_names: Vec<String>,
    pub sign_widths: Vec<i32>,
    pub sign_values: Vec<i32>,
    pub gdn_v_grouped: bool,
}

#[derive(Clone)]
pub(crate) struct HadamardTransform {
    pub(crate) signs: MxArray,
    pub(crate) block_size: i32,
    pub(crate) gdn_permutation: Option<(i32, i32, i32)>,
}

pub(crate) struct PrismHadamardRuntime {
    projections: HashMap<String, HadamardTransform>,
    embedding: HadamardTransform,
    signs: Vec<MxArray>,
}

fn weight_name_layer(name: &str) -> Result<Option<i32>> {
    let base = name.strip_suffix(WEIGHT_SUFFIX).ok_or_else(|| {
        Error::from_reason(format!(
            "prism_hadamard: weight name '{name}' must end with '{WEIGHT_SUFFIX}'"
        ))
    })?;
    if base == LM_HEAD_PREFIX {
        return Ok(None);
    }
    let rest = base.strip_prefix(LAYERS_PREFIX).ok_or_else(|| {
        Error::from_reason(format!(
            "prism_hadamard: unsupported weight name '{name}'; expected 'layers.<i>.<projection>.weight' or '{LM_HEAD_PREFIX}.weight'"
        ))
    })?;
    let (index, projection) = rest.split_once('.').ok_or_else(|| {
        Error::from_reason(format!(
            "prism_hadamard: unsupported weight name '{name}'; expected 'layers.<i>.<projection>.weight'"
        ))
    })?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::from_reason(format!(
            "prism_hadamard: weight name '{name}' has an invalid layer index '{index}'"
        )));
    }
    let index: i32 = index.parse().map_err(|_| {
        Error::from_reason(format!(
            "prism_hadamard: weight name '{name}' layer index '{index}' is out of range"
        ))
    })?;
    if !SUPPORTED_LAYER_PROJECTIONS.contains(&projection) {
        return Err(Error::from_reason(format!(
            "prism_hadamard: unsupported projection '{projection}' in weight name '{name}'"
        )));
    }
    Ok(Some(index))
}

fn required_dim(value: i32, label: &str) -> Result<i64> {
    if value <= 0 {
        return Err(Error::from_reason(format!(
            "prism_hadamard: {label} must be positive, got {value}"
        )));
    }
    Ok(i64::from(value))
}

fn checked_product(factors: &[i64], label: &str) -> Result<i64> {
    factors.iter().try_fold(1i64, |acc, &factor| {
        acc.checked_mul(factor)
            .ok_or_else(|| Error::from_reason(format!("prism_hadamard: {label} overflows i64")))
    })
}

fn expected_matrix_shape(prefix: &str, config: &Qwen3_5Config) -> Result<(i64, i64)> {
    let hidden = required_dim(config.hidden_size, "hidden_size")?;
    if prefix == "embedding" || prefix == LM_HEAD_PREFIX {
        return Ok((required_dim(config.vocab_size, "vocab_size")?, hidden));
    }
    let rest = prefix.strip_prefix(LAYERS_PREFIX).ok_or_else(|| {
        Error::from_reason(format!(
            "prism_hadamard: unsupported transform target '{prefix}'"
        ))
    })?;
    let (index, projection) = rest.split_once('.').ok_or_else(|| {
        Error::from_reason(format!(
            "prism_hadamard: unsupported transform target '{prefix}'"
        ))
    })?;
    let layer: usize = index.parse().map_err(|_| {
        Error::from_reason(format!("prism_hadamard: invalid layer index in '{prefix}'"))
    })?;
    let is_linear = config.is_linear_layer(layer);
    let require_kind = |wanted: bool| -> Result<()> {
        if is_linear != wanted {
            return Err(Error::from_reason(format!(
                "prism_hadamard: '{prefix}' belongs to the wrong attention kind; layer {layer} has is_linear_layer={is_linear}"
            )));
        }
        Ok(())
    };
    match projection {
        "mlp.gate_proj" | "mlp.up_proj" => Ok((
            required_dim(config.intermediate_size, "intermediate_size")?,
            hidden,
        )),
        "mlp.down_proj" => Ok((
            hidden,
            required_dim(config.intermediate_size, "intermediate_size")?,
        )),
        "linear_attn.in_proj_qkv" => {
            require_kind(true)?;
            let kh = required_dim(config.linear_num_key_heads, "linear_num_key_heads")?;
            let kd = required_dim(config.linear_key_head_dim, "linear_key_head_dim")?;
            let vh = required_dim(config.linear_num_value_heads, "linear_num_value_heads")?;
            let vd = required_dim(config.linear_value_head_dim, "linear_value_head_dim")?;
            let qkv = checked_product(&[2, kh, kd], "in_proj_qkv rows")?
                .checked_add(checked_product(&[vh, vd], "in_proj_qkv rows")?)
                .ok_or_else(|| {
                    Error::from_reason("prism_hadamard: in_proj_qkv rows overflow i64")
                })?;
            Ok((qkv, hidden))
        }
        "linear_attn.in_proj_z" => {
            require_kind(true)?;
            let vh = required_dim(config.linear_num_value_heads, "linear_num_value_heads")?;
            let vd = required_dim(config.linear_value_head_dim, "linear_value_head_dim")?;
            Ok((checked_product(&[vh, vd], "in_proj_z rows")?, hidden))
        }
        "linear_attn.out_proj" => {
            require_kind(true)?;
            let vh = required_dim(config.linear_num_value_heads, "linear_num_value_heads")?;
            let vd = required_dim(config.linear_value_head_dim, "linear_value_head_dim")?;
            Ok((hidden, checked_product(&[vh, vd], "out_proj rows")?))
        }
        "self_attn.q_proj" => {
            require_kind(false)?;
            let nh = required_dim(config.num_heads, "num_heads")?;
            let hd = required_dim(config.head_dim, "head_dim")?;
            Ok((checked_product(&[2, nh, hd], "q_proj rows")?, hidden))
        }
        "self_attn.k_proj" | "self_attn.v_proj" => {
            require_kind(false)?;
            let kvh = required_dim(config.num_kv_heads, "num_kv_heads")?;
            let hd = required_dim(config.head_dim, "head_dim")?;
            Ok((checked_product(&[kvh, hd], "kv_proj rows")?, hidden))
        }
        "self_attn.o_proj" => {
            require_kind(false)?;
            let nh = required_dim(config.num_heads, "num_heads")?;
            let hd = required_dim(config.head_dim, "head_dim")?;
            Ok((hidden, checked_product(&[nh, hd], "o_proj rows")?))
        }
        _ => Err(Error::from_reason(format!(
            "prism_hadamard: unsupported transform target '{prefix}'"
        ))),
    }
}

impl PrismHadamardConfig {
    pub(crate) fn from_config(raw: &serde_json::Value) -> Result<Option<Self>> {
        if raw
            .get("text_config")
            .and_then(|v| v.get(PRISM_HADAMARD_CONFIG_KEY))
            .is_some()
        {
            return Err(Error::from_reason(
                "prism_hadamard marker must be a top-level config key; a nested text_config.prism_hadamard marker is not supported",
            ));
        }
        let model_type = raw.get("model_type").and_then(|v| v.as_str());
        if model_type == Some("prism_hadamard_qwen35") {
            return Err(Error::from_reason(
                "model_type 'prism_hadamard_qwen35' is not a supported family; load the original GGUF whose converted config.json carries a top-level 'prism_hadamard' contract",
            ));
        }
        let Some(marker) = raw.get(PRISM_HADAMARD_CONFIG_KEY) else {
            return Ok(None);
        };
        if marker.is_null() {
            return Err(Error::from_reason(
                "prism_hadamard marker must be an object, not null",
            ));
        }
        if model_type != Some(EXPECTED_MODEL_TYPE) {
            return Err(Error::from_reason(format!(
                "prism_hadamard checkpoints require model_type '{EXPECTED_MODEL_TYPE}', got {model_type:?}; the 'prism_hadamard_qwen35' pack alias is not supported"
            )));
        }
        let config: Self = serde_json::from_value(marker.clone())
            .map_err(|e| Error::from_reason(format!("Invalid prism_hadamard config: {e}")))?;
        config.validate()?;
        Ok(Some(config))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.version != EXPECTED_VERSION {
            return Err(Error::from_reason(format!(
                "prism_hadamard: unsupported version {}; expected {EXPECTED_VERSION}",
                self.version
            )));
        }
        if self.block_size != EXPECTED_BLOCK_SIZE {
            return Err(Error::from_reason(format!(
                "prism_hadamard: unsupported block_size {}; expected {EXPECTED_BLOCK_SIZE}",
                self.block_size
            )));
        }
        if self.transform != EXPECTED_TRANSFORM {
            return Err(Error::from_reason(format!(
                "prism_hadamard: unsupported transform '{}'; expected '{EXPECTED_TRANSFORM}'",
                self.transform
            )));
        }
        if self.axis != EXPECTED_AXIS {
            return Err(Error::from_reason(format!(
                "prism_hadamard: unsupported axis '{}'; expected '{EXPECTED_AXIS}'",
                self.axis
            )));
        }
        if self.sign_mode != EXPECTED_SIGN_MODE {
            return Err(Error::from_reason(format!(
                "prism_hadamard: unsupported sign_mode '{}'; expected '{EXPECTED_SIGN_MODE}'",
                self.sign_mode
            )));
        }
        if !self.gdn_v_grouped {
            return Err(Error::from_reason(
                "prism_hadamard: gdn_v_grouped must be true; the ungrouped layout is not supported",
            ));
        }
        if self.weight_names.is_empty() {
            return Err(Error::from_reason(
                "prism_hadamard: weight_names must not be empty",
            ));
        }
        if self.inverse_weight_names.len() != 1
            || self.inverse_weight_names[0] != EMBEDDING_WEIGHT_NAME
        {
            return Err(Error::from_reason(format!(
                "prism_hadamard: inverse_weight_names must be exactly ['{EMBEDDING_WEIGHT_NAME}'], got {:?}",
                self.inverse_weight_names
            )));
        }
        let mut seen = HashSet::new();
        for name in self.weight_names.iter().chain(&self.inverse_weight_names) {
            if !seen.insert(name.as_str()) {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: duplicate or overlapping weight name '{name}'"
                )));
            }
        }
        for name in &self.weight_names {
            weight_name_layer(name)?;
        }
        let mut widths = HashSet::new();
        let mut total: i64 = 0;
        for &width in &self.sign_widths {
            if width <= 0 {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: sign width {width} must be positive"
                )));
            }
            if width % self.block_size != 0 {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: sign width {width} is not divisible by block_size {}",
                    self.block_size
                )));
            }
            if !widths.insert(width) {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: duplicate sign width {width}"
                )));
            }
            total = total
                .checked_add(i64::from(width))
                .ok_or_else(|| Error::from_reason("prism_hadamard: sign widths overflow"))?;
        }
        let declared = i64::try_from(self.sign_values.len()).map_err(|_| {
            Error::from_reason("prism_hadamard: sign_values length is out of range")
        })?;
        if total != declared {
            return Err(Error::from_reason(format!(
                "prism_hadamard: sign widths sum to {total} but sign_values has {declared} entries"
            )));
        }
        if let Some(&value) = self.sign_values.iter().find(|&&v| v != 1 && v != -1) {
            return Err(Error::from_reason(format!(
                "prism_hadamard: sign value {value} is invalid; only +/-1 is allowed"
            )));
        }
        Ok(())
    }

    fn shared_signs(
        &self,
        width: i64,
        label: &str,
        cache: &mut HashMap<i64, MxArray>,
        owned: &mut Vec<MxArray>,
    ) -> Result<MxArray> {
        if let Some(signs) = cache.get(&width) {
            return Ok(signs.clone());
        }
        let width_i32 = i32::try_from(width).map_err(|_| {
            Error::from_reason(format!(
                "prism_hadamard: '{label}' input width {width} is out of range"
            ))
        })?;
        let mut offset = 0usize;
        let mut found = None;
        for &w in &self.sign_widths {
            if w == width_i32 {
                found = Some(offset);
                break;
            }
            offset += w as usize;
        }
        let offset = found.ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: '{label}' input width {width} matches no declared sign width"
            ))
        })?;
        let end = offset + width_i32 as usize;
        let values: Vec<f32> = self.sign_values[offset..end]
            .iter()
            .map(|&v| v as f32)
            .collect();
        let signs = MxArray::from_float32(&values, &[width])?;
        cache.insert(width, signs.clone());
        owned.push(signs.clone());
        Ok(signs)
    }

    fn validated_transform(
        &self,
        name: &str,
        params: &HashMap<String, MxArray>,
        config: &Qwen3_5Config,
        default_quant: PerLayerQuant,
        overrides: &HashMap<String, PerLayerQuant>,
        sign_cache: &mut HashMap<i64, MxArray>,
        sign_arrays: &mut Vec<MxArray>,
    ) -> Result<HadamardTransform> {
        let prefix = name.strip_suffix(WEIGHT_SUFFIX).ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: weight name '{name}' must end with '{WEIGHT_SUFFIX}'"
            ))
        })?;
        let weight = params.get(name).ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' is absent from the checkpoint"
            ))
        })?;
        if weight.dtype()? != DType::Uint32 {
            return Err(Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' is dense {:?}; rotated weights must be packed Uint32",
                weight.dtype()?
            )));
        }
        let w_shape = weight.shape()?.to_vec();
        if w_shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' must be 2-D, got {w_shape:?}"
            )));
        }
        if w_shape.iter().any(|&axis| axis <= 0) {
            return Err(Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' has a non-positive packed axis {w_shape:?}"
            )));
        }
        let n = w_shape[0];
        let k = w_shape[1].checked_mul(PQ2_CODES_PER_WORD).ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' packed width overflows i64"
            ))
        })?;
        if k % i64::from(PQ2_GROUP_SIZE) != 0 {
            return Err(Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' logical width {k} is not a multiple of {PQ2_GROUP_SIZE}"
            )));
        }
        let (expected_n, expected_k) = expected_matrix_shape(prefix, config)?;
        if n != expected_n || k != expected_k {
            return Err(Error::from_reason(format!(
                "prism_hadamard: declared weight '{name}' has logical shape [{n}, {k}] but '{prefix}' requires [{expected_n}, {expected_k}]"
            )));
        }
        let groups = k / i64::from(PQ2_GROUP_SIZE);
        let plq = if prefix == "embedding" {
            overrides
                .get("embed_tokens")
                .copied()
                .unwrap_or(default_quant)
        } else {
            effective_plq_for(prefix, overrides, default_quant, None)
        };
        if plq.mode != PerLayerMode::Affine
            || plq.bits != PQ2_BITS
            || plq.group_size != PQ2_GROUP_SIZE
            || plq.input_amax.is_some()
        {
            return Err(Error::from_reason(format!(
                "prism_hadamard: '{prefix}' resolves to {}/{} {:?} (input_amax={:?}); rotated weights require affine {}/{}",
                plq.bits, plq.group_size, plq.mode, plq.input_amax, PQ2_BITS, PQ2_GROUP_SIZE
            )));
        }
        let scales = params.get(&format!("{prefix}.scales")).ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: '{prefix}.scales' is missing for declared weight '{name}'"
            ))
        })?;
        let biases = params.get(&format!("{prefix}.biases")).ok_or_else(|| {
            Error::from_reason(format!(
                "prism_hadamard: '{prefix}.biases' is missing for declared weight '{name}'"
            ))
        })?;
        for (sidecar, label) in [(scales, "scales"), (biases, "biases")] {
            if sidecar.dtype()? != DType::Float16 {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: '{prefix}.{label}' must be Float16, got {:?}",
                    sidecar.dtype()?
                )));
            }
            let sidecar_shape = sidecar.shape()?.to_vec();
            if sidecar_shape != vec![n, groups] {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: '{prefix}.{label}' must have shape [{n}, {groups}], got {sidecar_shape:?}"
                )));
            }
        }
        if scales.has_nan_or_inf()? {
            return Err(Error::from_reason(format!(
                "prism_hadamard: '{prefix}.scales' contains non-finite values"
            )));
        }
        let sum = scales
            .astype(DType::Float32)?
            .add(&biases.astype(DType::Float32)?)?;
        let max_abs = sum.abs()?.max(None, None)?;
        max_abs.eval();
        if max_abs.item_at_float32(0)? != 0.0 {
            return Err(Error::from_reason(format!(
                "prism_hadamard: '{prefix}.biases' must equal -scales for a PQ2_0 group"
            )));
        }
        let signs = self.shared_signs(k, prefix, sign_cache, sign_arrays)?;
        let gdn_permutation = if prefix.ends_with(GDN_OUT_PROJ_SUFFIX) {
            let value_heads = config.linear_num_value_heads;
            let key_heads = config.linear_num_key_heads;
            let head_dim = config.linear_value_head_dim;
            if value_heads <= 0 || key_heads <= 0 || head_dim <= 0 || value_heads % key_heads != 0 {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: invalid GDN geometry value_heads={value_heads} key_heads={key_heads} value_head_dim={head_dim} for '{prefix}'"
                )));
            }
            if i64::from(value_heads) * i64::from(head_dim) != k {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: '{prefix}' input width {k} must equal value_heads*value_head_dim ({value_heads}*{head_dim})"
                )));
            }
            Some((value_heads / key_heads, key_heads, head_dim))
        } else {
            None
        };
        Ok(HadamardTransform {
            signs,
            block_size: self.block_size,
            gdn_permutation,
        })
    }

    pub(crate) fn prepare(
        &self,
        params: &HashMap<String, MxArray>,
        config: &Qwen3_5Config,
        default_quant: PerLayerQuant,
        overrides: &HashMap<String, PerLayerQuant>,
    ) -> Result<PrismHadamardRuntime> {
        self.validate()?;
        if config.qwen35_gguf_gdn_layout.as_deref() != Some("tiled") {
            return Err(Error::from_reason(
                "prism_hadamard: requires qwen35_gguf_gdn_layout 'tiled'",
            ));
        }
        if config.tie_word_embeddings {
            return Err(Error::from_reason(
                "prism_hadamard: requires untied lm_head (tie_word_embeddings must be false)",
            ));
        }
        if config.n_mtp_layers > 0 {
            return Err(Error::from_reason(
                "prism_hadamard: MTP checkpoints are not supported",
            ));
        }
        let mut declared: HashSet<&str> = HashSet::new();
        for name in self.weight_names.iter().chain(&self.inverse_weight_names) {
            declared.insert(name.as_str());
            if !params.contains_key(name) {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: declared weight '{name}' is absent from the checkpoint"
                )));
            }
            if name.as_str() == EMBEDDING_WEIGHT_NAME {
                continue;
            }
            if let Some(layer) = weight_name_layer(name)?
                && layer >= config.num_layers
            {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: weight name '{name}' references layer {layer} but the model has {} layers",
                    config.num_layers
                )));
            }
        }
        for (key, value) in params {
            if key.ends_with(WEIGHT_SUFFIX)
                && value.dtype()? == DType::Uint32
                && !declared.contains(key.as_str())
            {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: packed weight '{key}' is not covered by the declared transform contract"
                )));
            }
        }
        let mut sign_cache: HashMap<i64, MxArray> = HashMap::new();
        let mut sign_arrays: Vec<MxArray> = Vec::new();
        let mut projections = HashMap::new();
        for name in &self.weight_names {
            let prefix = name.strip_suffix(WEIGHT_SUFFIX).ok_or_else(|| {
                Error::from_reason(format!(
                    "prism_hadamard: weight name '{name}' must end with '{WEIGHT_SUFFIX}'"
                ))
            })?;
            let transform = self.validated_transform(
                name,
                params,
                config,
                default_quant,
                overrides,
                &mut sign_cache,
                &mut sign_arrays,
            )?;
            projections.insert(prefix.to_string(), transform);
        }
        let embedding = self.validated_transform(
            EMBEDDING_WEIGHT_NAME,
            params,
            config,
            default_quant,
            overrides,
            &mut sign_cache,
            &mut sign_arrays,
        )?;
        Ok(PrismHadamardRuntime {
            projections,
            embedding,
            signs: sign_arrays,
        })
    }

    pub(crate) fn prepare_for_load(
        &self,
        params: &mut HashMap<String, MxArray>,
        config: &Qwen3_5Config,
        default_quant: PerLayerQuant,
        overrides: &HashMap<String, PerLayerQuant>,
        hoist_metadata: bool,
    ) -> Result<PrismHadamardRuntime> {
        let runtime = self.prepare(params, config, default_quant, overrides)?;
        if hoist_metadata {
            for prefix in runtime.projections.keys() {
                for suffix in ["scales", "biases"] {
                    let name = format!("{prefix}.{suffix}");
                    let metadata = params.get_mut(&name).ok_or_else(|| {
                        Error::from_reason(format!(
                            "prism_hadamard: validated metadata '{name}' disappeared during load"
                        ))
                    })?;
                    *metadata = metadata.astype(DType::Float32)?;
                    metadata.eval();
                }
            }
        }
        Ok(runtime)
    }
}

impl HadamardTransform {
    pub(crate) fn apply(&self, x: &MxArray, inverse: bool) -> Result<MxArray> {
        let shape = x.shape()?.to_vec();
        let dtype = x.dtype()?;
        if shape.is_empty() {
            return Err(Error::from_reason(
                "prism_hadamard: input must have at least one dimension",
            ));
        }
        let &last = shape.last().unwrap_or(&0);
        let sign_width = self.signs.shape_at(0)?;
        if last <= 0 || last != sign_width {
            return Err(Error::from_reason(format!(
                "prism_hadamard: input last dimension {last} must equal the sign width {sign_width}"
            )));
        }
        if last % i64::from(self.block_size) != 0 {
            return Err(Error::from_reason(format!(
                "prism_hadamard: input last dimension {last} is not divisible by block_size {}",
                self.block_size
            )));
        }
        match dtype {
            DType::Float32 | DType::Float16 | DType::BFloat16 => {}
            other => {
                return Err(Error::from_reason(format!(
                    "prism_hadamard: input dtype {other:?} must be floating-point"
                )));
            }
        }
        if inverse && self.gdn_permutation.is_some() {
            return Err(Error::from_reason(
                "prism_hadamard: inverse transform does not support a GDN permutation",
            ));
        }
        if share_hadamard_enabled()
            && let Some(cached) =
                HADAMARD_CACHE.with(|cache| cache.borrow().lookup(x, self, inverse))
        {
            return Ok(cached);
        }
        let mut y = x.astype(DType::Float32)?;
        if !inverse {
            if let Some((rep, nk, hd)) = self.gdn_permutation {
                y = y
                    .reshape(&[-1, i64::from(rep), i64::from(nk), i64::from(hd)])?
                    .transpose(Some(&[0, 2, 1, 3]))?
                    .reshape(&shape)?;
            }
            y = y.mul(&self.signs)?;
        }
        let blocked = y.reshape(&[-1, i64::from(self.block_size)])?;
        let handle = unsafe {
            mlx_sys::mlx_array_hadamard_transform(
                blocked.as_raw_ptr(),
                1.0 / (self.block_size as f32).sqrt(),
            )
        };
        y = MxArray::from_handle(handle, "prism_hadamard")?.reshape(&shape)?;
        if inverse {
            y = y.mul(&self.signs)?;
        }
        let result = y.astype(dtype)?;
        if share_hadamard_enabled() {
            HADAMARD_CACHE.with(|cache| cache.borrow_mut().store(x, self, inverse, &result));
        }
        Ok(result)
    }
}

impl PrismHadamardRuntime {
    pub(crate) fn projection(&self, prefix: &str) -> Option<HadamardTransform> {
        self.projections.get(prefix).cloned()
    }

    pub(crate) fn embedding(&self) -> HadamardTransform {
        self.embedding.clone()
    }

    pub(crate) fn arrays(&self) -> Vec<&MxArray> {
        self.signs.iter().collect()
    }

    pub(crate) fn nbytes(&self) -> u64 {
        self.signs.iter().fold(0u64, |total, signs| {
            total.saturating_add(signs.nbytes() as u64)
        })
    }
}

pub(crate) fn hoist_metadata_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_BONSAI_HOIST_METADATA").as_deref() == Ok("1"))
}

fn share_hadamard_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_BONSAI_SHARE_HADAMARD").as_deref() == Ok("1"))
}

type HadamardCacheKey = (usize, usize, i32, Option<(i32, i32, i32)>);

fn hadamard_cache_key(x: &MxArray, transform: &HadamardTransform) -> HadamardCacheKey {
    (
        x.as_raw_ptr() as usize,
        transform.signs.as_raw_ptr() as usize,
        transform.block_size,
        transform.gdn_permutation,
    )
}

struct HadamardCacheEntry {
    key: HadamardCacheKey,
    _input: MxArray,
    _signs: MxArray,
    result: MxArray,
}

#[derive(Default)]
struct HadamardCache {
    active: bool,
    last: Option<HadamardCacheEntry>,
}

impl HadamardCache {
    fn set_active(&mut self, active: bool) {
        self.last = None;
        self.active = active;
    }

    fn lookup(&self, x: &MxArray, transform: &HadamardTransform, inverse: bool) -> Option<MxArray> {
        if !self.active || inverse {
            return None;
        }
        let key = hadamard_cache_key(x, transform);
        self.last
            .as_ref()
            .filter(|entry| entry.key == key)
            .map(|entry| entry.result.clone())
    }

    fn store(
        &mut self,
        x: &MxArray,
        transform: &HadamardTransform,
        inverse: bool,
        result: &MxArray,
    ) {
        if self.active && !inverse {
            self.last = Some(HadamardCacheEntry {
                key: hadamard_cache_key(x, transform),
                _input: x.clone(),
                _signs: transform.signs.clone(),
                result: result.clone(),
            });
        }
    }
}

thread_local! {
    static HADAMARD_CACHE: std::cell::RefCell<HadamardCache> = std::cell::RefCell::new(HadamardCache::default());
}

pub(crate) struct HadamardDecodeScope(bool);

impl HadamardDecodeScope {
    pub(crate) fn enter(has_rotations: bool) -> Self {
        Self::with_enabled(has_rotations && share_hadamard_enabled())
    }

    fn with_enabled(enabled: bool) -> Self {
        if enabled {
            HADAMARD_CACHE.with(|cache| cache.borrow_mut().set_active(true));
        }
        Self(enabled)
    }
}

impl Drop for HadamardDecodeScope {
    fn drop(&mut self) {
        if self.0 {
            HADAMARD_CACHE.with(|cache| cache.borrow_mut().set_active(false));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign_array(values: &[i32]) -> MxArray {
        MxArray::from_float32(
            &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &[values.len() as i64],
        )
        .unwrap()
    }

    fn nontrivial_signs(width: usize) -> Vec<i32> {
        (0..width)
            .map(|i| if (i * 5 + 1) % 11 < 5 { -1 } else { 1 })
            .collect()
    }

    fn det_input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect()
    }

    fn wht_block(v: &mut [f32], scale: f32) {
        let n = v.len();
        let mut h = 1usize;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let a = v[j];
                    let b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for x in v.iter_mut() {
            *x *= scale;
        }
    }

    fn reference_apply(
        x: &[f32],
        rows: usize,
        width: usize,
        signs: &[f32],
        block: usize,
        gdn: Option<(usize, usize, usize)>,
        inverse: bool,
    ) -> Vec<f32> {
        let scale = 1.0 / (block as f32).sqrt();
        let mut out = vec![0f32; x.len()];
        for r in 0..rows {
            let mut y = x[r * width..(r + 1) * width].to_vec();
            if !inverse {
                if let Some((rep, nk, hd)) = gdn {
                    let mut p = vec![0f32; width];
                    for b in 0..nk {
                        for a in 0..rep {
                            for c in 0..hd {
                                p[b * rep * hd + a * hd + c] = y[a * nk * hd + b * hd + c];
                            }
                        }
                    }
                    y = p;
                }
                for (v, &s) in y.iter_mut().zip(signs) {
                    *v *= s;
                }
            }
            for blk in y.chunks_exact_mut(block) {
                wht_block(blk, scale);
            }
            if inverse {
                for (v, &s) in y.iter_mut().zip(signs) {
                    *v *= s;
                }
            }
            out[r * width..(r + 1) * width].copy_from_slice(&y);
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn transform_for(signs: &[i32], block: i32, gdn: Option<(i32, i32, i32)>) -> HadamardTransform {
        HadamardTransform {
            signs: sign_array(signs),
            block_size: block,
            gdn_permutation: gdn,
        }
    }

    fn valid_config() -> PrismHadamardConfig {
        PrismHadamardConfig {
            version: 1,
            block_size: 1024,
            transform: "normalized-sylvester-walsh-hadamard".to_string(),
            axis: "input-last-dimension".to_string(),
            sign_mode: "explicit".to_string(),
            weight_names: vec![
                "layers.0.linear_attn.in_proj_qkv.weight".to_string(),
                "layers.0.linear_attn.out_proj.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            inverse_weight_names: vec!["embedding.weight".to_string()],
            sign_widths: vec![1024, 2048],
            sign_values: (0..3072).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
            gdn_v_grouped: true,
        }
    }

    #[test]
    fn prism_hadamard_apply_matches_scalar_butterfly() {
        for width in [1024usize, 5120, 6144, 17408] {
            for rows in [1usize, 17] {
                let signs_i = nontrivial_signs(width);
                let signs_f: Vec<f32> = signs_i.iter().map(|&v| v as f32).collect();
                let transform = transform_for(&signs_i, 1024, None);
                let input = det_input(rows * width);
                let x = MxArray::from_float32(&input, &[rows as i64, width as i64]).unwrap();
                for inverse in [false, true] {
                    let expected =
                        reference_apply(&input, rows, width, &signs_f, 1024, None, inverse);
                    let got = transform.apply(&x, inverse).unwrap();
                    let got = got.to_float32().unwrap();
                    let diff = max_abs_diff(got.as_ref(), &expected);
                    assert!(
                        diff <= 1e-5,
                        "width {width} rows {rows} inverse {inverse}: max abs diff {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn prism_hadamard_apply_forward_then_inverse_restores_input() {
        let width = 1024usize;
        let transform = transform_for(&nontrivial_signs(width), 1024, None);
        let input = det_input(3 * width);
        let x = MxArray::from_float32(&input, &[3, width as i64]).unwrap();
        let roundtrip = transform
            .apply(&transform.apply(&x, false).unwrap(), true)
            .unwrap();
        let diff = max_abs_diff(roundtrip.to_float32().unwrap().as_ref(), &input);
        assert!(diff <= 1e-5, "forward+inverse roundtrip diff {diff}");
    }

    #[test]
    fn prism_hadamard_apply_groups_gdn_value_heads_three_to_one() {
        let (rep, nk, hd) = (3usize, 16usize, 128usize);
        let width = rep * nk * hd;
        let head_src: Vec<usize> = (0..nk * rep).map(|oh| (oh % rep) * nk + oh / rep).collect();
        assert_eq!(&head_src[..6], &[0, 16, 32, 1, 17, 33]);
        let signs_i = nontrivial_signs(width);
        let signs_f: Vec<f32> = signs_i.iter().map(|&v| v as f32).collect();
        let transform = transform_for(&signs_i, 1024, Some((3, 16, 128)));
        let rows = 2usize;
        let mut input = vec![0f32; rows * width];
        for r in 0..rows {
            for h in 0..nk * rep {
                for c in 0..hd {
                    input[r * width + h * hd + c] = (h + 1 + r * 64) as f32;
                }
            }
        }
        let expected = reference_apply(
            &input,
            rows,
            width,
            &signs_f,
            1024,
            Some((rep, nk, hd)),
            false,
        );
        let x = MxArray::from_float32(&input, &[rows as i64, width as i64]).unwrap();
        let got = transform.apply(&x, false).unwrap();
        let diff = max_abs_diff(got.to_float32().unwrap().as_ref(), &expected);
        assert!(diff <= 1e-5, "GDN grouped permutation diverged: {diff}");
    }

    #[test]
    fn prism_hadamard_apply_preserves_rank_and_casts_back() {
        let width = 1024i64;
        let transform = transform_for(&nontrivial_signs(width as usize), 1024, None);
        let x2 = MxArray::from_float32(&det_input(2 * width as usize), &[2, width]).unwrap();
        let y2 = transform.apply(&x2, false).unwrap();
        assert_eq!(y2.shape().unwrap().to_vec(), vec![2, width]);
        assert_eq!(y2.dtype().unwrap(), DType::Float32);
        let x3 = MxArray::from_float32(&det_input(6 * width as usize), &[2, 3, width])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let y3 = transform.apply(&x3, true).unwrap();
        assert_eq!(y3.shape().unwrap().to_vec(), vec![2, 3, width]);
        assert_eq!(y3.dtype().unwrap(), DType::Float16);
    }

    #[test]
    fn prism_hadamard_apply_rejects_bad_inputs() {
        let width = 1024usize;
        let transform = transform_for(&nontrivial_signs(width), 1024, None);
        let wrong_width = MxArray::from_float32(&det_input(2048), &[1, 2048]).unwrap();
        assert!(transform.apply(&wrong_width, false).is_err());
        let non_multiple = MxArray::from_float32(&det_input(1536), &[1, 1536]).unwrap();
        let unaligned = transform_for(&nontrivial_signs(1536), 1024, None);
        assert!(unaligned.apply(&non_multiple, false).is_err());
        let ints = MxArray::from_int32(&[0i32; 1024], &[1, 1024]).unwrap();
        assert!(transform.apply(&ints, false).is_err());
        let gdn = transform_for(&nontrivial_signs(width), 1024, Some((3, 16, 128)));
        let x = MxArray::from_float32(&det_input(width), &[1, 1024]).unwrap();
        assert!(gdn.apply(&x, true).is_err());
    }

    #[test]
    fn prism_hadamard_decode_cache_matches_full_transform_identity() {
        let values = det_input(1024);
        let x = MxArray::from_float32(&values, &[1, 1024]).unwrap();
        let other_x = MxArray::from_float32(&values, &[1, 1024]).unwrap();
        let signs = nontrivial_signs(1024);
        let transform = transform_for(&signs, 1024, None);
        let result = transform.apply(&x, false).unwrap();
        let mut cache = HadamardCache::default();
        cache.set_active(true);
        cache.store(&x, &transform, false, &result);
        let hit = cache.lookup(&x.clone(), &transform.clone(), false).unwrap();
        assert_eq!(hit.as_raw_ptr(), result.as_raw_ptr());
        assert_eq!(
            hit.to_float32().unwrap().as_ref(),
            result.to_float32().unwrap().as_ref()
        );
        assert!(cache.lookup(&other_x, &transform, false).is_none());
        assert!(
            cache
                .lookup(&x, &transform_for(&signs, 1024, None), false)
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &x,
                    &HadamardTransform {
                        block_size: 512,
                        ..transform.clone()
                    },
                    false
                )
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &x,
                    &HadamardTransform {
                        gdn_permutation: Some((2, 4, 128)),
                        ..transform.clone()
                    },
                    false
                )
                .is_none()
        );
        assert!(cache.lookup(&x, &transform, true).is_none());
    }

    #[test]
    fn prism_hadamard_decode_cache_does_not_retain_outside_step() {
        let x = MxArray::from_float32(&det_input(1024), &[1, 1024]).unwrap();
        let transform = transform_for(&nontrivial_signs(1024), 1024, None);
        let result = transform.apply(&x, false).unwrap();
        let mut cache = HadamardCache::default();
        cache.store(&x, &transform, false, &result);
        assert!(cache.last.is_none());
        cache.set_active(true);
        cache.store(&x, &transform, true, &result);
        assert!(cache.last.is_none());
        cache.store(&x, &transform, false, &result);
        assert!(cache.lookup(&x, &transform, false).is_some());
        cache.set_active(true);
        assert!(cache.last.is_none());
        cache.store(&x, &transform, false, &result);
        cache.set_active(false);
        assert!(!cache.active);
        assert!(cache.last.is_none());
        assert!(cache.lookup(&x, &transform, false).is_none());
    }

    #[test]
    fn prism_hadamard_decode_scope_clears_cache_on_error() {
        let x = MxArray::from_float32(&det_input(1024), &[1, 1024]).unwrap();
        let transform = transform_for(&nontrivial_signs(1024), 1024, None);
        let result = transform.apply(&x, false).unwrap();
        let outcome: Result<()> = {
            let _scope = HadamardDecodeScope::with_enabled(true);
            HADAMARD_CACHE.with(|cache| cache.borrow_mut().store(&x, &transform, false, &result));
            HADAMARD_CACHE.with(|cache| assert!(cache.borrow().last.is_some()));
            Err(Error::from_reason("scope cleanup probe"))
        };
        assert!(outcome.is_err());
        HADAMARD_CACHE.with(|cache| {
            let cache = cache.borrow();
            assert!(!cache.active);
            assert!(cache.last.is_none());
        });
    }

    #[test]
    fn prism_hadamard_from_config_parses_and_gates() {
        let valid = serde_json::to_value(valid_config()).unwrap();
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "prism_hadamard": valid,
        });
        let parsed = PrismHadamardConfig::from_config(&raw).unwrap();
        assert_eq!(parsed.unwrap().sign_widths, vec![1024, 2048]);

        let bare = serde_json::json!({ "model_type": "qwen3_5" });
        assert!(PrismHadamardConfig::from_config(&bare).unwrap().is_none());

        let null_marker = serde_json::json!({
            "model_type": "qwen3_5",
            "prism_hadamard": null,
        });
        assert!(PrismHadamardConfig::from_config(&null_marker).is_err());

        let nested = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": { "prism_hadamard": valid },
        });
        assert!(PrismHadamardConfig::from_config(&nested).is_err());

        let nested_with_root = serde_json::json!({
            "model_type": "qwen3_5",
            "prism_hadamard": valid,
            "text_config": { "prism_hadamard": valid },
        });
        assert!(
            PrismHadamardConfig::from_config(&nested_with_root).is_err(),
            "a nested marker must be rejected even when a root marker exists"
        );

        let alias_without_marker = serde_json::json!({
            "model_type": "prism_hadamard_qwen35",
        });
        assert!(
            PrismHadamardConfig::from_config(&alias_without_marker).is_err(),
            "the prism_hadamard_qwen35 alias must not bypass the contract gate"
        );

        for model_type in ["qwen3", "prism_hadamard_qwen35", "qwen35moe"] {
            let wrong = serde_json::json!({
                "model_type": model_type,
                "prism_hadamard": valid,
            });
            assert!(
                PrismHadamardConfig::from_config(&wrong).is_err(),
                "model_type {model_type} must be rejected"
            );
        }

        for (label, mutate) in [
            (
                "version",
                Box::new(|c: &mut PrismHadamardConfig| c.version = 2)
                    as Box<dyn Fn(&mut PrismHadamardConfig)>,
            ),
            (
                "sign value",
                Box::new(|c: &mut PrismHadamardConfig| c.sign_values[0] = 0),
            ),
        ] {
            let mut contract = valid_config();
            mutate(&mut contract);
            let raw = serde_json::json!({
                "model_type": "qwen3_5",
                "prism_hadamard": serde_json::to_value(&contract).unwrap(),
            });
            assert!(
                PrismHadamardConfig::from_config(&raw).is_err(),
                "a contract failing '{label}' validation must be rejected at parse time"
            );
        }

        let mut extra = serde_json::to_value(valid_config()).unwrap();
        extra
            .as_object_mut()
            .unwrap()
            .insert("extra_field".to_string(), serde_json::json!(1));
        let unknown = serde_json::json!({
            "model_type": "qwen3_5",
            "prism_hadamard": extra,
        });
        assert!(PrismHadamardConfig::from_config(&unknown).is_err());
    }

    #[test]
    fn prism_hadamard_validate_accepts_fixture_and_rejects_mutations() {
        valid_config().validate().unwrap();
        let cases: Vec<(&str, Box<dyn Fn(&mut PrismHadamardConfig)>)> = vec![
            ("version", Box::new(|c| c.version = 2)),
            ("block_size", Box::new(|c| c.block_size = 512)),
            ("transform", Box::new(|c| c.transform = "other".to_string())),
            ("axis", Box::new(|c| c.axis = "rows".to_string())),
            (
                "sign_mode",
                Box::new(|c| c.sign_mode = "implicit".to_string()),
            ),
            ("gdn_v_grouped", Box::new(|c| c.gdn_v_grouped = false)),
            ("empty weights", Box::new(|c| c.weight_names.clear())),
            (
                "empty inverse",
                Box::new(|c| c.inverse_weight_names.clear()),
            ),
            (
                "wrong inverse",
                Box::new(|c| c.inverse_weight_names = vec!["lm_head.weight".to_string()]),
            ),
            (
                "duplicate name",
                Box::new(|c| {
                    c.weight_names
                        .push("layers.0.mlp.gate_proj.weight".to_string());
                    c.weight_names
                        .push("layers.0.mlp.gate_proj.weight".to_string());
                }),
            ),
            (
                "overlap with inverse",
                Box::new(|c| c.weight_names.push("embedding.weight".to_string())),
            ),
            (
                "unsupported projection",
                Box::new(|c| {
                    c.weight_names
                        .push("layers.0.linear_attn.in_proj_a.weight".to_string())
                }),
            ),
            (
                "bad layer index",
                Box::new(|c| {
                    c.weight_names
                        .push("layers.x.mlp.gate_proj.weight".to_string())
                }),
            ),
            (
                "missing weight suffix",
                Box::new(|c| c.weight_names.push("layers.0.mlp.gate_proj".to_string())),
            ),
            (
                "duplicate width",
                Box::new(|c| {
                    c.sign_widths = vec![1024, 1024, 1024];
                }),
            ),
            ("zero width", Box::new(|c| c.sign_widths[0] = 0)),
            ("negative width", Box::new(|c| c.sign_widths[1] = -2048)),
            ("unaligned width", Box::new(|c| c.sign_widths[1] = 2047)),
            (
                "missing signs",
                Box::new(|c| {
                    c.sign_values.pop();
                }),
            ),
            (
                "trailing signs",
                Box::new(|c| {
                    c.sign_values.push(1);
                }),
            ),
            ("zero sign", Box::new(|c| c.sign_values[7] = 0)),
            ("two sign", Box::new(|c| c.sign_values[9] = 2)),
        ];
        for (label, mutate) in cases {
            let mut config = valid_config();
            mutate(&mut config);
            assert!(
                config.validate().is_err(),
                "mutation '{label}' must be rejected"
            );
        }
    }

    fn mini_qwen35_config() -> Qwen3_5Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 1024,
            "num_layers": 4,
            "num_heads": 8,
            "num_kv_heads": 1,
            "intermediate_size": 1024,
            "rms_norm_eps": 1e-6,
            "head_dim": 128,
            "tie_word_embeddings": false,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "qwen35_gguf_gdn_layout": "tiled"
        }))
        .unwrap()
    }

    fn pq2_quant() -> PerLayerQuant {
        PerLayerQuant {
            bits: 2,
            group_size: 128,
            mode: PerLayerMode::Affine,
            input_amax: None,
        }
    }

    fn packed_triplet(params: &mut HashMap<String, MxArray>, prefix: &str, n: i64, k: i64) {
        let words = (k / 16) as usize;
        let groups = (k / 128) as usize;
        let weights: Vec<u32> = (0..(n as usize) * words)
            .map(|i| (i as u32).wrapping_mul(0x9E3779B1))
            .collect();
        params.insert(
            format!("{prefix}.weight"),
            MxArray::from_uint32(&weights, &[n, k / 16]).unwrap(),
        );
        let scales: Vec<u16> = (0..(n as usize) * groups)
            .map(|i| half::f16::from_f32(0.5 + (i % 7) as f32 * 0.25).to_bits())
            .collect();
        let biases: Vec<u16> = scales.iter().map(|&s| s ^ 0x8000).collect();
        params.insert(
            format!("{prefix}.scales"),
            MxArray::from_float16(&scales, &[n, groups as i64]).unwrap(),
        );
        params.insert(
            format!("{prefix}.biases"),
            MxArray::from_float16(&biases, &[n, groups as i64]).unwrap(),
        );
    }

    fn mini_prism_params() -> (PrismHadamardConfig, HashMap<String, MxArray>) {
        let mut config = valid_config();
        config.weight_names = vec![
            "layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            "layers.0.linear_attn.out_proj.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        config.sign_widths = vec![1024, 6144];
        config.sign_values = (0..7168).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let mut params = HashMap::new();
        packed_triplet(&mut params, "layers.0.linear_attn.in_proj_qkv", 10240, 1024);
        packed_triplet(&mut params, "layers.0.linear_attn.out_proj", 1024, 6144);
        packed_triplet(&mut params, "lm_head", 8, 1024);
        packed_triplet(&mut params, "embedding", 8, 1024);
        params.insert(
            "layers.0.linear_attn.in_proj_b.weight".to_string(),
            MxArray::from_float32(&[0.0f32; 16], &[4, 4]).unwrap(),
        );
        (config, params)
    }

    #[test]
    fn prism_hadamard_load_preparation_accounts_for_hoisted_metadata() {
        use crate::models::quantized_linear::QuantizedLinear;
        for enabled in [false, true] {
            let (config, mut params) = mini_prism_params();
            let original = params.clone();
            let bytes = |params: &HashMap<String, MxArray>| {
                params.values().fold(0u64, |total, array| {
                    total.saturating_add(array.nbytes() as u64)
                })
            };
            let before = bytes(&params);
            let runtime = config
                .prepare_for_load(
                    &mut params,
                    &mini_qwen35_config(),
                    pq2_quant(),
                    &HashMap::new(),
                    enabled,
                )
                .unwrap();
            let expected_extra = if enabled { 524_544 } else { 0 };
            assert_eq!(
                bytes(&params).saturating_add(runtime.nbytes()),
                before
                    .saturating_add(runtime.nbytes())
                    .saturating_add(expected_extra)
            );
            let metadata: HashSet<String> = config
                .weight_names
                .iter()
                .flat_map(|name| {
                    let prefix = name.strip_suffix(".weight").unwrap();
                    [format!("{prefix}.scales"), format!("{prefix}.biases")]
                })
                .collect();
            for (name, source) in &original {
                let actual = &params[name];
                assert_eq!(
                    actual.shape().unwrap().as_ref(),
                    source.shape().unwrap().as_ref()
                );
                if enabled && metadata.contains(name) {
                    assert_eq!(actual.dtype().unwrap(), DType::Float32, "{name}");
                    assert_eq!(
                        actual.to_float32().unwrap().as_ref(),
                        source
                            .astype(DType::Float32)
                            .unwrap()
                            .to_float32()
                            .unwrap()
                            .as_ref(),
                        "{name}"
                    );
                } else {
                    assert_eq!(actual.as_raw_ptr(), source.as_raw_ptr(), "{name}");
                }
            }
            if enabled {
                let linear = QuantizedLinear::new(
                    params["lm_head.weight"].clone(),
                    params["lm_head.scales"].clone(),
                    Some(params["lm_head.biases"].clone()),
                    None,
                    128,
                    2,
                    "affine".to_string(),
                )
                .with_hadamard(runtime.projection("lm_head"))
                .unwrap();
                assert_eq!(linear.get_scales().dtype().unwrap(), DType::Float32);
                assert_eq!(
                    linear.get_scales().nbytes(),
                    params["lm_head.scales"].nbytes()
                );
                assert_eq!(
                    linear.get_biases().unwrap().nbytes(),
                    params["lm_head.biases"].nbytes()
                );
            }
        }
    }

    #[test]
    fn prism_hadamard_load_preparation_validates_before_hoisting() {
        let (config, mut params) = mini_prism_params();
        params.insert(
            "lm_head.scales".to_string(),
            params["lm_head.scales"].astype(DType::Float32).unwrap(),
        );
        let original = params.clone();
        let error = match config.prepare_for_load(
            &mut params,
            &mini_qwen35_config(),
            pq2_quant(),
            &HashMap::new(),
            true,
        ) {
            Ok(_) => panic!("non-FP16 source metadata must still be rejected"),
            Err(error) => error,
        };
        assert!(error.reason.contains("must be Float16"), "{}", error.reason);
        for (name, source) in original {
            assert_eq!(params[&name].as_raw_ptr(), source.as_raw_ptr(), "{name}");
        }
    }

    #[test]
    fn prism_hadamard_prepare_installs_transforms_and_sign_arrays() {
        let (config, params) = mini_prism_params();
        let runtime = config
            .prepare(&params, &mini_qwen35_config(), pq2_quant(), &HashMap::new())
            .unwrap();
        assert!(
            runtime
                .projection("layers.0.linear_attn.in_proj_qkv")
                .is_some()
        );
        let out_proj = runtime.projection("layers.0.linear_attn.out_proj").unwrap();
        assert_eq!(out_proj.gdn_permutation, Some((3, 16, 128)));
        assert!(
            runtime
                .projection("layers.0.linear_attn.in_proj_z")
                .is_none()
        );
        let embedding = runtime.embedding();
        assert_eq!(embedding.gdn_permutation, None);
        assert_eq!(runtime.arrays().len(), 2);
        assert_eq!(runtime.nbytes(), (1024 + 6144) * 4);
    }

    #[test]
    fn prism_hadamard_prepare_rejects_unsupported_checkpoints() {
        let (config, params) = mini_prism_params();
        let qconfig = mini_qwen35_config();
        let quant = pq2_quant();
        let none = HashMap::new();

        let mut dense = params.clone();
        dense.insert(
            "layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 4 * 1024], &[4, 1024]).unwrap(),
        );
        assert!(
            config.prepare(&dense, &qconfig, quant, &none).is_err(),
            "a dense declared weight must be rejected"
        );

        let mut extra_packed = params.clone();
        packed_triplet(&mut extra_packed, "layers.0.mlp.down_proj", 4, 1024);
        assert!(
            config
                .prepare(&extra_packed, &qconfig, quant, &none)
                .is_err(),
            "a packed weight outside the manifest must be rejected"
        );

        let mut missing = params.clone();
        missing.remove("embedding.weight");
        assert!(
            config.prepare(&missing, &qconfig, quant, &none).is_err(),
            "an absent declared weight must be rejected"
        );

        let mut wrong_bias = params.clone();
        wrong_bias.insert(
            "lm_head.biases".to_string(),
            MxArray::from_float16(&[0u16; 8 * 8], &[8, 8]).unwrap(),
        );
        assert!(
            config.prepare(&wrong_bias, &qconfig, quant, &none).is_err(),
            "biases other than -scales must be rejected"
        );

        let mut nan_scales = params.clone();
        nan_scales.insert(
            "lm_head.scales".to_string(),
            MxArray::from_float16(&[half::f16::NAN.to_bits(); 8 * 8], &[8, 8]).unwrap(),
        );
        assert!(
            config.prepare(&nan_scales, &qconfig, quant, &none).is_err(),
            "non-finite scales must be rejected"
        );

        let mut tied = qconfig.clone();
        tied.tie_word_embeddings = true;
        assert!(
            config.prepare(&params, &tied, quant, &none).is_err(),
            "tied embeddings must be rejected"
        );

        let mut mtp = qconfig.clone();
        mtp.n_mtp_layers = 1;
        assert!(
            config.prepare(&params, &mtp, quant, &none).is_err(),
            "MTP checkpoints must be rejected"
        );

        let mut non_tiled = qconfig.clone();
        non_tiled.qwen35_gguf_gdn_layout = Some("interleaved".to_string());
        assert!(
            config.prepare(&params, &non_tiled, quant, &none).is_err(),
            "a non-tiled GDN layout must be rejected"
        );

        let mut wrong_quant = pq2_quant();
        wrong_quant.bits = 4;
        wrong_quant.group_size = 32;
        assert!(
            config
                .prepare(&params, &qconfig, wrong_quant, &none)
                .is_err(),
            "a non-2/128 quant profile must be rejected"
        );

        let mut tiny = qconfig.clone();
        tiny.num_layers = 0;
        let mut out_of_range = config.clone();
        out_of_range.weight_names = vec!["layers.7.mlp.gate_proj.weight".to_string()];
        let mut layer_params = HashMap::new();
        packed_triplet(&mut layer_params, "layers.7.mlp.gate_proj", 4, 1024);
        packed_triplet(&mut layer_params, "embedding", 8, 1024);
        out_of_range.sign_widths = vec![1024];
        out_of_range.sign_values = vec![1; 1024];
        assert!(
            out_of_range
                .prepare(&layer_params, &tiny, quant, &none)
                .is_err(),
            "a declared layer beyond the model depth must be rejected"
        );
    }

    #[test]
    fn prism_hadamard_linear_and_embedding_execute_transforms() -> Result<()> {
        use crate::models::quantized_linear::QuantizedLinear;
        use crate::nn::Embedding;
        let mut params = HashMap::new();
        packed_triplet(&mut params, "probe", 32, 1024);
        let w = &params["probe.weight"];
        let s = &params["probe.scales"];
        let b = &params["probe.biases"];
        let transform = transform_for(&nontrivial_signs(1024), 1024, None);
        let make_linear = || {
            QuantizedLinear::new(
                w.clone(),
                s.clone(),
                Some(b.clone()),
                None,
                128,
                2,
                "affine".to_string(),
            )
        };
        let base = make_linear();
        let mut rotated = make_linear().with_hadamard(Some(transform.clone()))?;
        for batch in [1, 17] {
            let x = MxArray::from_float32(&det_input(batch * 1024), &[batch as i64, 1024])?;
            let expected = base.forward(&transform.apply(&x, false)?)?.to_float32()?;
            let actual = rotated.forward(&x)?.to_float32()?;
            assert_eq!(actual.len(), expected.len());
            assert!(max_abs_diff(&actual, &expected) <= 1e-5);
        }
        let x = MxArray::from_float32(&det_input(17 * 1024), &[17, 1024])?;
        let expected = rotated.forward(&x)?;
        assert!(rotated.finalize_packed_q_gate_block(4, 4)?);
        let order: Vec<i32> = (0..2)
            .flat_map(|part| {
                (0..4).flat_map(move |head| (0..4).map(move |dim| head * 8 + part * 4 + dim))
            })
            .collect();
        let expected = expected
            .take(&MxArray::from_int32(&order, &[32])?, 1)?
            .to_float32()?;
        let actual = rotated.forward(&x)?.to_float32()?;
        assert!(max_abs_diff(&actual, &expected) <= 1e-5);

        let mut plain = Embedding::new_uninitialized(32, 1024)?;
        plain.load_quantized_packed(w, s, Some(b), 128, 2, "affine")?;
        let mut embedding = plain.clone();
        embedding.set_hadamard(transform.clone())?;
        let ids = MxArray::from_uint32(&[0, 7, 31], &[3])?;
        let expected = transform
            .apply(&plain.forward(&ids)?, true)?
            .astype(DType::Float32)?
            .to_float32()?;
        for candidate in [&embedding, &embedding.clone()] {
            let actual = candidate
                .forward(&ids)?
                .astype(DType::Float32)?
                .to_float32()?;
            assert_eq!(actual.as_ref(), expected.as_ref());
            assert_eq!(candidate.packed_full_dequant_calls(), 0);
        }
        let expected = plain
            .as_linear(&transform.apply(&x, false)?)?
            .to_float32()?;
        let actual = embedding.as_linear(&x)?.to_float32()?;
        assert!(max_abs_diff(&actual, &expected) <= 1e-5);
        assert_eq!(embedding.packed_full_dequant_calls(), 0);
        embedding.load_quantized_packed(w, s, Some(b), 128, 2, "affine")?;
        assert!(!embedding.has_hadamard());
        assert_eq!(
            embedding
                .forward(&ids)?
                .astype(DType::Float32)?
                .to_float32()?
                .as_ref(),
            plain
                .forward(&ids)?
                .astype(DType::Float32)?
                .to_float32()?
                .as_ref()
        );
        Ok(())
    }

    #[test]
    fn prism_hadamard_hoisted_projection_promotes_half_inputs() -> Result<()> {
        use crate::models::quantized_linear::QuantizedLinear;
        let mut params = HashMap::new();
        packed_triplet(&mut params, "probe", 32, 1024);
        let transform = transform_for(&nontrivial_signs(1024), 1024, None);
        let make = |dtype| -> Result<QuantizedLinear> {
            QuantizedLinear::new(
                params["probe.weight"].clone(),
                params["probe.scales"].astype(dtype)?,
                Some(params["probe.biases"].astype(dtype)?),
                None,
                128,
                2,
                "affine".to_string(),
            )
            .with_hadamard(Some(transform.clone()))
        };
        // FP32 metadata stands in for the hoisted loader path; whether the
        // env flag itself is set decides what `with_hadamard` leaves behind
        // for the FP16 operand, so the expected baseline dtype follows it.
        let hoisted = make(DType::Float32)?;
        let unhoisted = make(DType::Float16)?;
        // The promoted pipeline is transform-then-widen: the transform keeps
        // its residual-dtype output contract, so the exact reference applies
        // it on the 16-bit input, widens, and runs a plain FP32-metadata QMM.
        let base32 = QuantizedLinear::new(
            params["probe.weight"].clone(),
            params["probe.scales"].astype(DType::Float32)?,
            Some(params["probe.biases"].astype(DType::Float32)?),
            None,
            128,
            2,
            "affine".to_string(),
        );
        for dtype in [DType::Float16, DType::BFloat16] {
            let x = MxArray::from_float32(&det_input(4 * 1024), &[4, 1024])?.astype(dtype)?;
            // A real Prism load feeds the transform 16-bit residual-stream
            // activations; hoisted metadata must promote them, not error.
            let out = hoisted.forward(&x)?;
            assert_eq!(out.dtype()?, DType::Float32);
            let expected = base32.forward(&transform.apply(&x, false)?.astype(DType::Float32)?)?;
            assert_eq!(out.to_float32()?.as_ref(), expected.to_float32()?.as_ref());
            let baseline = unhoisted.forward(&x)?;
            let baseline_dtype = if hoist_metadata_enabled() {
                DType::Float32
            } else {
                dtype
            };
            assert_eq!(baseline.dtype()?, baseline_dtype);
        }
        Ok(())
    }

    fn prism_case(
        declared: &[(&str, i64, i64)],
    ) -> (PrismHadamardConfig, HashMap<String, MxArray>) {
        let mut config = valid_config();
        config.weight_names = declared
            .iter()
            .map(|(name, _, _)| format!("{name}.weight"))
            .collect();
        let mut widths: Vec<i64> = declared.iter().map(|(_, _, k)| *k).collect();
        widths.push(1024);
        widths.sort_unstable();
        widths.dedup();
        config.sign_widths = widths.iter().map(|&w| w as i32).collect();
        config.sign_values = (0..widths.iter().sum::<i64>())
            .map(|i| if i % 3 == 0 { -1 } else { 1 })
            .collect();
        let mut params = HashMap::new();
        for (name, n, k) in declared {
            packed_triplet(&mut params, name, *n, *k);
        }
        packed_triplet(&mut params, "embedding", 8, 1024);
        (config, params)
    }

    #[test]
    fn prism_hadamard_prepare_covers_all_projection_kinds() {
        let declared: [(&str, i64, i64); 14] = [
            ("layers.0.linear_attn.in_proj_qkv", 10240, 1024),
            ("layers.0.linear_attn.in_proj_z", 6144, 1024),
            ("layers.0.linear_attn.out_proj", 1024, 6144),
            ("layers.0.mlp.gate_proj", 1024, 1024),
            ("layers.0.mlp.up_proj", 1024, 1024),
            ("layers.0.mlp.down_proj", 1024, 1024),
            ("layers.3.self_attn.q_proj", 2048, 1024),
            ("layers.3.self_attn.k_proj", 128, 1024),
            ("layers.3.self_attn.v_proj", 128, 1024),
            ("layers.3.self_attn.o_proj", 1024, 1024),
            ("layers.3.mlp.gate_proj", 1024, 1024),
            ("layers.3.mlp.up_proj", 1024, 1024),
            ("layers.3.mlp.down_proj", 1024, 1024),
            ("lm_head", 8, 1024),
        ];
        let (config, params) = prism_case(&declared);
        let runtime = config
            .prepare(&params, &mini_qwen35_config(), pq2_quant(), &HashMap::new())
            .unwrap();
        for (name, _, _) in &declared {
            assert!(
                runtime.projection(name).is_some(),
                "a transform must be installed for '{name}'"
            );
        }
        for absent in [
            "layers.0.linear_attn.in_proj_b",
            "layers.0.linear_attn.in_proj_a",
            "layers.0.linear_attn.in_proj_qkvz",
        ] {
            assert!(
                runtime.projection(absent).is_none(),
                "no transform may be installed for '{absent}'"
            );
        }
        assert_eq!(
            runtime
                .projection("layers.0.linear_attn.out_proj")
                .unwrap()
                .gdn_permutation,
            Some((3, 16, 128))
        );
    }

    #[test]
    fn prism_hadamard_prepare_rejects_shape_mutations() {
        let qconfig = mini_qwen35_config();
        let quant = pq2_quant();
        let none = HashMap::new();

        let (config, params) = prism_case(&[
            ("layers.0.self_attn.q_proj", 2048, 1024),
            ("lm_head", 8, 1024),
        ]);
        assert!(
            config.prepare(&params, &qconfig, quant, &none).is_err(),
            "a self_attn projection on a linear-attention layer must be rejected"
        );

        let (config, params) = prism_case(&[
            ("layers.3.linear_attn.out_proj", 1024, 6144),
            ("lm_head", 8, 1024),
        ]);
        assert!(
            config.prepare(&params, &qconfig, quant, &none).is_err(),
            "a linear_attn projection on a full-attention layer must be rejected"
        );

        let (config, params) = prism_case(&[
            ("layers.0.linear_attn.in_proj_qkv", 8, 1024),
            ("lm_head", 8, 1024),
        ]);
        assert!(
            config.prepare(&params, &qconfig, quant, &none).is_err(),
            "a wrong output row count must be rejected"
        );

        let (config, params) = prism_case(&[
            ("layers.0.linear_attn.out_proj", 1024, 1024),
            ("lm_head", 8, 1024),
        ]);
        assert!(
            config.prepare(&params, &qconfig, quant, &none).is_err(),
            "a wrong logical input width must be rejected"
        );

        let (config, mut params) = prism_case(&[
            ("layers.0.linear_attn.in_proj_qkv", 10240, 1024),
            ("lm_head", 8, 1024),
        ]);
        packed_triplet(&mut params, "layers.0.linear_attn.in_proj_qkv", 10240, 1040);
        assert!(
            config.prepare(&params, &qconfig, quant, &none).is_err(),
            "a logical width that is not a multiple of 128 must be rejected"
        );

        let mut bad_geometry = qconfig.clone();
        bad_geometry.linear_num_value_heads = 0;
        let (config, params) = mini_prism_params();
        assert!(
            config
                .prepare(&params, &bad_geometry, quant, &none)
                .is_err(),
            "non-positive head geometry must be rejected"
        );

        let mut bad_heads = qconfig.clone();
        bad_heads.num_heads = 0;
        let (config, params) = prism_case(&[
            ("layers.3.self_attn.q_proj", 2048, 1024),
            ("lm_head", 8, 1024),
        ]);
        assert!(
            config.prepare(&params, &bad_heads, quant, &none).is_err(),
            "non-positive attention head geometry must be rejected"
        );
    }

    #[test]
    fn prism_hadamard_prepare_rejects_embedding_quant_override() {
        let (config, params) = mini_prism_params();
        let qconfig = mini_qwen35_config();
        let quant = pq2_quant();
        let conflicting = HashMap::from([(
            "embed_tokens".to_string(),
            PerLayerQuant {
                bits: 4,
                group_size: 64,
                mode: PerLayerMode::Affine,
                input_amax: None,
            },
        )]);
        assert!(
            config
                .prepare(&params, &qconfig, quant, &conflicting)
                .is_err(),
            "an embed_tokens override other than affine 2/128 must be rejected"
        );
        let matching = HashMap::from([("embed_tokens".to_string(), pq2_quant())]);
        assert!(
            config.prepare(&params, &qconfig, quant, &matching).is_ok(),
            "an embed_tokens override matching affine 2/128 must be accepted"
        );
    }
}
