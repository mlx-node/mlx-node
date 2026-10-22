use super::config::invalid;
use crate::{
    array::{DType, MxArray},
    models::{
        quant_dispatch::{self, KQuantLayout, PerLayerMode, PerLayerQuant},
        quantized_linear::{LinearProj, QuantizedLinear},
        rerank_checkpoint::{UniqueMap, validate_header},
    },
    nn::{Embedding, LayerNorm, Linear},
    utils::safetensors::load_safetensors_lazy,
};
use napi::Result;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
};

pub(crate) fn read_weights(dir: &Path) -> Result<HashMap<String, MxArray>> {
    let index_path = dir.join("model.safetensors.index.json");
    let mut files = BTreeSet::new();
    let expected = if index_path.exists() {
        #[derive(Deserialize)]
        struct Index {
            weight_map: UniqueMap<String>,
        }
        let index: Index =
            serde_json::from_str(&std::fs::read_to_string(index_path).map_err(invalid)?)
                .map_err(invalid)?;
        let map = index.weight_map.0;
        if map.is_empty() {
            return Err(invalid("empty weight_map"));
        }
        let mut expected = HashMap::new();
        for (key, file) in map {
            let file = file.as_str();
            if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                return Err(invalid("invalid shard path"));
            }
            files.insert(file.to_owned());
            expected.insert(key.clone(), file.to_owned());
        }
        Some(expected)
    } else {
        let found: Vec<_> = ["model.safetensors", "weights.safetensors"]
            .into_iter()
            .filter(|f| dir.join(f).exists())
            .collect();
        if found.len() != 1 {
            return Err(invalid("expected one checkpoint or a safetensors index"));
        }
        files.insert(found[0].to_owned());
        None
    };
    let mut out = HashMap::new();
    for file in files {
        validate_header(&dir.join(&file))?;
        for (key, value) in load_safetensors_lazy(dir.join(&file))? {
            if let Some(expected) = &expected
                && expected.get(&key) != Some(&file)
            {
                return Err(invalid(format!("tensor {key} does not match shard index")));
            }
            if out.insert(key.clone(), value).is_some() {
                return Err(invalid(format!("duplicate tensor {key}")));
            }
        }
    }
    if let Some(expected) = expected
        && let Some(missing) = expected.keys().find(|key| !out.contains_key(*key))
    {
        return Err(invalid(format!("missing index tensor {missing}")));
    }
    Ok(out)
}
fn validate_quantization(q: PerLayerQuant, context: &str) -> Result<()> {
    if q.bits != 4
        || !matches!(q.mode, PerLayerMode::Affine | PerLayerMode::Mxfp4)
        || q.input_amax.is_some()
        || q.layout != KQuantLayout::RowMajor
        || q.group_size <= 0
        || (q.mode == PerLayerMode::Mxfp4 && q.group_size != 32)
    {
        return Err(invalid(format!("unsupported quantization for {context}")));
    }
    Ok(())
}

pub(crate) struct Weights {
    pub tensors: HashMap<String, MxArray>,
    overrides: HashMap<String, PerLayerQuant>,
    default: PerLayerQuant,
    has_quant_config: bool,
}
impl Weights {
    pub fn new(dir: &Path, raw: &Value) -> Result<Self> {
        Self::from_tensors(read_weights(dir)?, raw)
    }
    pub fn from_tensors(tensors: HashMap<String, MxArray>, raw: &Value) -> Result<Self> {
        let block = quant_dispatch::select_quantization_block(raw)?;
        let mut declared = BTreeSet::new();
        if let Some(object) = block.and_then(Value::as_object) {
            for (name, value) in object {
                let module = quant_dispatch::normalize_per_layer_key(name);
                let is_module = module.starts_with("roberta.") || module.starts_with("classifier.");
                if is_module && !value.as_object().is_some_and(|v| v.contains_key("bits")) {
                    return Err(invalid(format!(
                        "invalid quantization declaration for {module}: expected an object with bits"
                    )));
                }
                if value.is_object() && !declared.insert(module) {
                    return Err(invalid(format!(
                        "duplicate quantization declaration {name}"
                    )));
                }
            }
        }
        let (bits, group_size, mode, overrides) =
            quant_dispatch::parse_quant_settings(block, 4, 64)?;
        let default = PerLayerQuant {
            bits,
            group_size,
            mode: mode.unwrap_or(PerLayerMode::Affine),
            input_amax: None,
            layout: KQuantLayout::RowMajor,
        };
        // Declared formats are part of this backend's contract even for modules
        // retained as dense. Do not silently accept another family's format.
        validate_quantization(default, "default quantization")?;
        for (name, quant) in &overrides {
            validate_quantization(*quant, name)?;
        }
        Ok(Self {
            tensors,
            overrides,
            default,
            has_quant_config: block.is_some(),
        })
    }
    pub fn tensor(&self, key: &str) -> Result<&MxArray> {
        self.tensors
            .get(key)
            .ok_or_else(|| invalid(format!("missing tensor {key}")))
    }
    fn dense(&self, key: &str, shape: &[i64]) -> Result<&MxArray> {
        let t = self.tensor(key)?;
        quant_dispatch::ensure_dense_weight_floating(key, t)?;
        if t.shape()?.as_ref() != shape {
            return Err(invalid(format!("invalid shape for {key}")));
        }
        Ok(t)
    }
    fn matrix(&self, base: &str, rows: i64, columns: i64) -> Result<()> {
        let w = self.tensor(&format!("{base}.weight"))?;
        let shape = w.shape()?;
        if self.tensors.contains_key(&format!("{base}.scales")) {
            if !self.has_quant_config {
                return Err(invalid(
                    "packed weights require explicit quantization metadata",
                ));
            }
            let q = quant_dispatch::effective_plq_for(base, &self.overrides, self.default, None);
            validate_quantization(q, base)?;
            if columns % i64::from(q.group_size) != 0 || columns % 8 != 0 {
                return Err(invalid(format!("invalid quantization geometry for {base}")));
            }
            if shape.as_ref() != [rows, columns / 8] || w.dtype()? != DType::Uint32 {
                return Err(invalid(format!("invalid packed shape/dtype for {base}")));
            }
            let scales = self.tensor(&format!("{base}.scales"))?;
            if scales.shape()?.as_ref() != [rows, columns / i64::from(q.group_size)] {
                return Err(invalid(format!("invalid scales for {base}")));
            }
            if q.mode == PerLayerMode::Affine {
                let biases = self.tensor(&format!("{base}.biases"))?;
                if biases.shape()?.as_ref() != scales.shape()?.as_ref() {
                    return Err(invalid(format!("invalid biases for {base}")));
                }
                quant_dispatch::ensure_dense_weight_floating(base, scales)?;
                quant_dispatch::ensure_dense_weight_floating(base, biases)?;
            } else if scales.dtype()? != DType::Uint8
                || self.tensors.contains_key(&format!("{base}.biases"))
            {
                return Err(invalid(format!("invalid MXFP4 sidecars for {base}")));
            }
        } else {
            quant_dispatch::ensure_dense_weight_floating(base, w)?;
            if shape.as_ref() != [rows, columns]
                || self.tensors.contains_key(&format!("{base}.biases"))
            {
                return Err(invalid(format!(
                    "invalid dense shape or orphan sidecar for {base}"
                )));
            }
        }
        Ok(())
    }

    pub fn linear(&self, base: &str, rows: i64, columns: i64) -> Result<LinearProj> {
        self.matrix(base, rows, columns)?;
        let bias = self.dense(&format!("{base}.bias"), &[rows])?;
        let weight = self.tensor(&format!("{base}.weight"))?;
        if let Some(scales) = self.tensors.get(&format!("{base}.scales")) {
            let q = quant_dispatch::effective_plq_for(base, &self.overrides, self.default, None);
            Ok(LinearProj::Quantized(QuantizedLinear::new(
                weight.clone(),
                scales.clone(),
                self.tensors.get(&format!("{base}.biases")).cloned(),
                Some(bias.clone()),
                q.group_size,
                q.bits,
                if q.mode == PerLayerMode::Mxfp4 {
                    "mxfp4"
                } else {
                    "affine"
                }
                .into(),
            )))
        } else {
            Ok(LinearProj::Standard(Linear::from_weights(
                weight,
                Some(bias),
            )?))
        }
    }
    pub fn norm(&self, base: &str, dim: i64, eps: f64) -> Result<LayerNorm> {
        LayerNorm::from_weights(
            self.dense(&format!("{base}.weight"), &[dim])?,
            Some(self.dense(&format!("{base}.bias"), &[dim])?),
            Some(eps),
        )
    }
    pub fn embedding(&self, base: &str, rows: i64, dim: i64, packed: bool) -> Result<Embedding> {
        let mut e = Embedding::new_uninitialized(rows as u32, dim as u32)?;
        if packed {
            self.matrix(base, rows, dim)?;
            quant_dispatch::load_embedding_affine_or_bf16(
                &mut e,
                &self.tensors,
                base,
                &self.overrides,
                self.default,
                "XLM-R reranker",
            )?;
        } else {
            e.set_weight(self.dense(&format!("{base}.weight"), &[rows, dim])?)?;
        }
        Ok(e)
    }
    /// Conversion uses the same required tensor contract before writing output.
    pub fn validate_config(&self, c: &super::config::Config) -> Result<()> {
        self.validate_keys(c.num_hidden_layers)?;
        let h = c.hidden_size;
        self.matrix("roberta.embeddings.word_embeddings", c.vocab_size, h)?;
        self.dense(
            "roberta.embeddings.position_embeddings.weight",
            &[c.max_position_embeddings as i64, h],
        )?;
        self.dense(
            "roberta.embeddings.token_type_embeddings.weight",
            &[c.type_vocab_size, h],
        )?;
        self.norm("roberta.embeddings.LayerNorm", h, c.layer_norm_eps)?;
        for i in 0..c.num_hidden_layers {
            let base = format!("roberta.encoder.layer.{i}");
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                self.linear(&format!("{base}.{name}"), h, h)?;
            }
            self.linear(
                &format!("{base}.intermediate.dense"),
                c.intermediate_size,
                h,
            )?;
            self.linear(&format!("{base}.output.dense"), h, c.intermediate_size)?;
            for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
                self.norm(&format!("{base}.{name}"), h, c.layer_norm_eps)?;
            }
        }
        self.linear("classifier.dense", h, h)?;
        self.linear("classifier.out_proj", 1, h)?;
        Ok(())
    }
    pub fn compute_dtype(&self, raw: &Value) -> Result<DType> {
        let dtype = self
            .tensor("roberta.embeddings.position_embeddings.weight")?
            .dtype()?;
        let label = match dtype {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            _ => return Err(invalid("unsupported compute dtype")),
        };
        if raw
            .get("reranker")
            .and_then(|m| m.get("computeDtype"))
            .is_some_and(|v| v != label)
        {
            return Err(invalid("compute dtype metadata conflicts with weights"));
        }
        for (key, tensor) in &self.tensors {
            // Affine sidecars may use higher precision; QMM restores activation
            // dtype at each projection boundary. Dense modules share one dtype.
            if key.ends_with(".weight") || key.ends_with(".bias") {
                let actual = tensor.dtype()?;
                if actual != DType::Uint32 && actual != dtype {
                    return Err(invalid(format!(
                        "inconsistent dense compute dtype for {key}"
                    )));
                }
            }
        }
        Ok(dtype)
    }
    pub fn validate_keys(&self, layers: usize) -> Result<()> {
        let mut allowed = BTreeSet::new();
        let mut matrix = |base: String, bias: bool, packed: bool| {
            allowed.insert(format!("{base}.weight"));
            if bias {
                allowed.insert(format!("{base}.bias"));
            }
            if packed {
                allowed.insert(format!("{base}.scales"));
                allowed.insert(format!("{base}.biases"));
            }
        };
        matrix("roberta.embeddings.word_embeddings".into(), false, true);
        matrix(
            "roberta.embeddings.position_embeddings".into(),
            false,
            false,
        );
        matrix(
            "roberta.embeddings.token_type_embeddings".into(),
            false,
            false,
        );
        matrix("roberta.embeddings.LayerNorm".into(), true, false);
        for i in 0..layers {
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ] {
                matrix(format!("roberta.encoder.layer.{i}.{name}"), true, true);
            }
            for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
                matrix(format!("roberta.encoder.layer.{i}.{name}"), true, false);
            }
        }
        matrix("classifier.dense".into(), true, true);
        matrix("classifier.out_proj".into(), true, true);
        if let Some(key) = self.tensors.keys().find(|k| !allowed.contains(*k)) {
            return Err(invalid(format!("unexpected tensor {key}")));
        }
        for base in self.overrides.keys() {
            if !self.tensors.contains_key(&format!("{base}.weight")) {
                return Err(invalid(format!("unknown quantization module {base}")));
            }
        }
        Ok(())
    }
    pub fn formats(&self) -> Vec<&'static str> {
        let mut formats = BTreeSet::new();
        for key in self
            .tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".weight"))
        {
            if self.tensors.contains_key(&format!("{key}.scales")) {
                let q = quant_dispatch::effective_plq_for(key, &self.overrides, self.default, None);
                formats.insert(if q.mode == PerLayerMode::Mxfp4 {
                    "mxfp4"
                } else {
                    "affine-4bit"
                });
            } else {
                formats.insert("dense");
            }
        }
        formats.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_declarations_are_rejected_without_packed_tensors() {
        for quantization in [
            serde_json::json!({"mode":"mxfp8","bits":8,"group_size":32}),
            serde_json::json!({"mode":"affine","bits":8,"group_size":64}),
            serde_json::json!({"classifier.dense":{"mode":"mxfp8","bits":8,"group_size":32}}),
            serde_json::json!({"classifier.dense":{}}),
            serde_json::json!({"classifier.dense":false}),
            serde_json::json!({"roberta.embeddings.word_embeddings":{}}),
            serde_json::json!({"roberta.encoder.layer.0.attention.self.query":false}),
        ] {
            assert!(
                Weights::from_tensors(
                    HashMap::new(),
                    &serde_json::json!({"quantization":quantization})
                )
                .is_err()
            );
        }
        for quantization in [
            serde_json::json!({"mode":"affine","bits":4,"group_size":64}),
            serde_json::json!({"mode":"mxfp4","bits":4,"group_size":32}),
        ] {
            assert!(
                Weights::from_tensors(
                    HashMap::new(),
                    &serde_json::json!({"quantization":quantization})
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn packed_linear_keeps_additive_bias_including_single_output_head() {
        for mode in [PerLayerMode::Affine, PerLayerMode::Mxfp4] {
            for rows in [1, 64] {
                let name = if mode == PerLayerMode::Affine {
                    "affine"
                } else {
                    "mxfp4"
                };
                let data: Vec<f32> = (0..rows * 64)
                    .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
                    .collect();
                let dense = MxArray::from_float32(&data, &[rows, 64]).unwrap();
                let mut q = std::ptr::null_mut();
                let mut s = std::ptr::null_mut();
                let mut b = std::ptr::null_mut();
                let mode_c = std::ffi::CString::new(name).unwrap();
                assert!(unsafe {
                    mlx_sys::mlx_quantize(
                        dense.as_raw_ptr(),
                        32,
                        4,
                        mode_c.as_ptr(),
                        &mut q,
                        &mut s,
                        &mut b,
                    )
                });
                let mut tensors = HashMap::new();
                tensors.insert(
                    "classifier.out_proj.weight".into(),
                    MxArray::from_handle(q, "q").unwrap(),
                );
                tensors.insert(
                    "classifier.out_proj.scales".into(),
                    MxArray::from_handle(s, "s").unwrap(),
                );
                if !b.is_null() {
                    tensors.insert(
                        "classifier.out_proj.biases".into(),
                        MxArray::from_handle(b, "b").unwrap(),
                    );
                }
                let bias = MxArray::from_float32(
                    &(0..rows).map(|i| 0.2 + i as f32 * 0.03).collect::<Vec<_>>(),
                    &[rows],
                )
                .unwrap();
                tensors.insert("classifier.out_proj.bias".into(), bias.clone());
                let weights = Weights {
                    tensors,
                    overrides: HashMap::new(),
                    default: PerLayerQuant {
                        bits: 4,
                        group_size: 32,
                        mode,
                        input_amax: None,
                        layout: KQuantLayout::RowMajor,
                    },
                    has_quant_config: true,
                };
                let projection = weights.linear("classifier.out_proj", rows, 64).unwrap();
                let x = MxArray::from_float32(
                    &(0..128).map(|i| (i % 13) as f32 * 0.02).collect::<Vec<_>>(),
                    &[2, 64],
                )
                .unwrap();
                let w = projection
                    .take_dequantized_rows(&(0..rows as u32).collect::<Vec<_>>())
                    .unwrap();
                let expected = x
                    .matmul(&w.transpose(Some(&[1, 0])).unwrap())
                    .unwrap()
                    .add(&bias)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let actual = projection.forward(&x).unwrap().to_float32().unwrap();
                for (a, b) in actual.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-4, "{name} bias lost: {a} != {b}");
                }
            }
        }
    }
}
