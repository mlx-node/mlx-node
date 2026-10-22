use super::config::invalid;
use crate::{
    array::{DType, MxArray},
    models::{
        quant_dispatch::{self, KQuantLayout, PerLayerMode, PerLayerQuant},
        quantized_linear::LinearProj,
        rerank_checkpoint::{UniqueMap, validate_header},
    },
    nn::{Embedding, Linear, RMSNorm},
    utils::safetensors::load_safetensors_lazy,
};
use napi::Result;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
};

pub(crate) fn canonical_key(key: &str) -> String {
    let key = key.strip_prefix("model.").unwrap_or(key);
    for (from, to) in [
        ("linear1.", "projector.0."),
        ("linear2.", "projector.2."),
        ("projector.layers.0.", "projector.0."),
        ("projector.layers.2.", "projector.2."),
    ] {
        if let Some(tail) = key.strip_prefix(from) {
            return format!("{to}{tail}");
        }
    }
    key.to_owned()
}

fn canonical_projection_key(name: &str) -> String {
    let normalized = quant_dispatch::normalize_per_layer_key(name);
    canonical_key(&format!("{normalized}.weight"))
        .trim_end_matches(".weight")
        .to_owned()
}

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
        let mut canonical = BTreeSet::new();
        for key in map.keys() {
            if !canonical.insert(canonical_key(key)) {
                return Err(invalid(format!("duplicate index tensor alias {key}")));
            }
        }
        for file in map.values() {
            if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                return Err(invalid("invalid shard path"));
            }
            files.insert(file.to_owned());
        }
        Some(map)
    } else {
        let candidates: Vec<_> = ["model.safetensors", "weights.safetensors"]
            .into_iter()
            .filter(|f| dir.join(f).exists())
            .collect();
        if candidates.len() != 1 {
            return Err(invalid("expected one checkpoint or a safetensors index"));
        }
        files.insert(candidates[0].into());
        None
    };
    let projector_in_index = files.contains("projector.safetensors");
    let mut out = HashMap::new();
    let mut canonical = BTreeSet::new();
    for file in files {
        validate_header(&dir.join(&file))?;
        for (key, value) in load_safetensors_lazy(dir.join(&file))? {
            if let Some(expected) = &expected
                && expected.get(&key) != Some(&file)
            {
                return Err(invalid(format!("tensor {key} does not match shard index")));
            }
            if !canonical.insert(canonical_key(&key)) {
                return Err(invalid(format!("duplicate tensor {key}")));
            }
            out.insert(key, value);
        }
    }
    if let Some(expected) = expected
        && let Some(missing) = expected.keys().find(|k| !out.contains_key(*k))
    {
        return Err(invalid(format!("index tensor missing: {missing}")));
    }
    if !projector_in_index && dir.join("projector.safetensors").exists() {
        validate_header(&dir.join("projector.safetensors"))?;
        for (key, value) in load_safetensors_lazy(dir.join("projector.safetensors"))? {
            let normalized = canonical_key(&key);
            if !normalized.starts_with("projector.") || !canonical.insert(normalized) {
                return Err(invalid(format!(
                    "unexpected or conflicting projector tensor {key}"
                )));
            }
            out.insert(key, value);
        }
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
        // The shared parser normalizes wrapper prefixes. Detect aliases before
        // parsing so a second spelling cannot silently replace an override.
        let mut declared = BTreeSet::new();
        if let Some(object) = block.and_then(Value::as_object) {
            for (name, value) in object {
                let module = canonical_projection_key(name);
                let is_module = matches!(module.as_str(), "embed_tokens" | "norm" | "lm_head")
                    || module.starts_with("layers.")
                    || module.starts_with("projector.");
                if is_module && !value.as_object().is_some_and(|v| v.contains_key("bits")) {
                    return Err(invalid(format!(
                        "invalid quantization declaration for {module}: expected an object with bits"
                    )));
                }
                if value.is_object() && !declared.insert(module.clone()) {
                    return Err(invalid(format!(
                        "duplicate quantization declaration for {module}"
                    )));
                }
            }
        }
        let (bits, group_size, mode, overrides) =
            quant_dispatch::parse_quant_settings(block, 4, 64)?;
        let overrides: HashMap<_, _> = overrides
            .into_iter()
            .map(|(k, v)| (canonical_projection_key(&k), v))
            // Causal-LM exports may quantize their unused vocabulary head with
            // another format. Neither that head nor its override belongs to ranking.
            .filter(|(k, _)| k != "lm_head")
            .collect();
        let default = PerLayerQuant {
            bits,
            group_size,
            mode: mode.unwrap_or(PerLayerMode::Affine),
            input_amax: None,
            layout: KQuantLayout::RowMajor,
        };
        // A declaration is part of the storage contract even when that module
        // remains dense. Unsupported declarations must not become silently active
        // only after a checkpoint starts carrying packed sidecars.
        validate_quantization(default, "default quantization")?;
        for (name, quant) in &overrides {
            validate_quantization(*quant, name)?;
        }
        let mut canonical = HashMap::new();
        for (name, tensor) in tensors {
            let key = canonical_key(&name);
            if key.starts_with("lm_head.") {
                continue;
            }
            if canonical.insert(key.clone(), tensor).is_some() {
                return Err(invalid(format!("duplicate tensor {key}")));
            }
        }
        Ok(Self {
            tensors: canonical,
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
    fn matrix(&self, base: &str, rows: i64, columns: i64) -> Result<()> {
        if self.tensors.contains_key(&format!("{base}.bias")) {
            return Err(invalid(format!("unsupported bias tensor for {base}")));
        }
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
        if self.tensors.contains_key(&format!("{base}.scales")) {
            let q = quant_dispatch::build_non_moe_ql(
                &self.tensors,
                base,
                &self.overrides,
                self.default,
                "Jina reranker",
            )?
            .ok_or_else(|| invalid(format!("invalid projection {base}")))?;
            Ok(LinearProj::Quantized(q))
        } else {
            Ok(LinearProj::Standard(Linear::from_weights(
                self.tensor(&format!("{base}.weight"))?,
                None,
            )?))
        }
    }
    fn norm_weight(&self, base: &str, dim: i64) -> Result<&MxArray> {
        if ["bias", "scales", "biases"]
            .iter()
            .any(|suffix| self.tensors.contains_key(&format!("{base}.{suffix}")))
        {
            return Err(invalid(format!("unsupported norm sidecar for {base}")));
        }
        let w = self.tensor(&format!("{base}.weight"))?;
        if w.shape()?.as_ref() != [dim] {
            return Err(invalid(format!("invalid norm shape {base}")));
        }
        quant_dispatch::ensure_dense_weight_floating(base, w)?;
        Ok(w)
    }
    pub fn norm(&self, base: &str, dim: i64, eps: f64) -> Result<RMSNorm> {
        RMSNorm::from_weight(self.norm_weight(base, dim)?, Some(eps))
    }
    /// Load and conversion share this complete, metadata-only tensor contract.
    /// Inspecting shapes and storage dtypes neither constructs random modules
    /// nor materializes/dequantizes packed checkpoint weights.
    pub fn validate_config(&self, c: &super::config::Config) -> Result<()> {
        let h = c.hidden_size;
        let d = c.head_dim;
        let inter = c.intermediate_size;
        let pd = self.output_dim("projector.0")?;
        let output = self.output_dim("projector.2")?;
        let mut matrices = vec![
            ("embed_tokens".to_owned(), c.vocab_size, h),
            ("projector.0".to_owned(), pd, h),
            ("projector.2".to_owned(), output, pd),
        ];
        let mut norms = vec![("norm".to_owned(), h)];
        for i in 0..c.num_hidden_layers {
            for (name, rows, columns) in [
                ("self_attn.q_proj", c.num_attention_heads * d, h),
                ("self_attn.k_proj", c.num_key_value_heads * d, h),
                ("self_attn.v_proj", c.num_key_value_heads * d, h),
                ("self_attn.o_proj", h, c.num_attention_heads * d),
                ("mlp.gate_proj", inter, h),
                ("mlp.up_proj", inter, h),
                ("mlp.down_proj", h, inter),
            ] {
                matrices.push((format!("layers.{i}.{name}"), rows, columns));
            }
            for (name, dim) in [
                ("input_layernorm", h),
                ("post_attention_layernorm", h),
                ("self_attn.q_norm", d),
                ("self_attn.k_norm", d),
            ] {
                norms.push((format!("layers.{i}.{name}"), dim));
            }
        }
        let mut allowed = BTreeSet::new();
        for (base, rows, columns) in &matrices {
            self.matrix(base, *rows, *columns)?;
            for suffix in ["weight", "scales", "biases"] {
                allowed.insert(format!("{base}.{suffix}"));
            }
        }
        for (base, dim) in &norms {
            self.norm_weight(base, *dim)?;
            allowed.insert(format!("{base}.weight"));
        }
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
    pub fn embedding(&self, vocab: i64, hidden: i64) -> Result<Embedding> {
        self.matrix("embed_tokens", vocab, hidden)?;
        let mut e = Embedding::new_uninitialized(vocab as u32, hidden as u32)?;
        quant_dispatch::load_embedding_affine_or_bf16(
            &mut e,
            &self.tensors,
            "embed_tokens",
            &self.overrides,
            self.default,
            "Jina reranker",
        )?;
        Ok(e)
    }
    pub fn output_dim(&self, base: &str) -> Result<i64> {
        let shape = self.tensor(&format!("{base}.weight"))?.shape()?;
        if shape.len() != 2 || shape[0] <= 0 {
            return Err(invalid(format!("invalid projector {base}")));
        }
        Ok(shape[0])
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
    fn conflicting_quantization_aliases_fail_before_tensor_validation() {
        let raw = serde_json::json!({"quantization": {
            "bits": 4, "group_size": 64,
            "model.layers.0.self_attn.q_proj": {"bits": 4, "group_size": 64},
            "layers.0.self_attn.q_proj": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        }});
        let error = Weights::from_tensors(HashMap::new(), &raw).err().unwrap();
        assert!(error.reason.contains("duplicate quantization declaration"));
    }
    #[test]
    fn unsupported_quantization_declarations_fail_even_without_packed_tensors() {
        for quantization in [
            serde_json::json!({"mode":"mxfp8","bits":8,"group_size":32}),
            serde_json::json!({"mode":"affine","bits":8,"group_size":64}),
            serde_json::json!({"projector.0":{"mode":"mxfp8","bits":8,"group_size":32}}),
            serde_json::json!({"projector.0":false}),
            serde_json::json!({"projector.0":{}}),
        ] {
            assert!(
                Weights::from_tensors(
                    HashMap::new(),
                    &serde_json::json!({"quantization": quantization})
                )
                .is_err()
            );
        }
    }
    #[test]
    fn unused_lm_head_override_is_not_a_ranking_quantization_declaration() {
        let weights = Weights::from_tensors(
            HashMap::new(),
            &serde_json::json!({"quantization":{
                "mode":"affine", "bits":4, "group_size":64,
                "model.lm_head":{"mode":"mxfp8","bits":8,"group_size":32}
            }}),
        )
        .unwrap();
        assert!(weights.overrides.is_empty());
    }
    fn fixture() -> Weights {
        Weights {
            tensors: HashMap::new(),
            overrides: HashMap::new(),
            default: PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Affine,
                input_amax: None,
                layout: KQuantLayout::RowMajor,
            },
            has_quant_config: true,
        }
    }
    #[test]
    fn missing_dense_and_incomplete_packed_weights_fail_before_dispatch() {
        let mut w = fixture();
        assert!(w.linear("q", 32, 32).is_err());
        w.tensors.insert(
            "q.weight".into(),
            MxArray::ones(&[32, 4], Some(DType::Uint32)).unwrap(),
        );
        assert!(w.linear("q", 32, 32).is_err());
        w.tensors.insert(
            "q.scales".into(),
            MxArray::ones(&[32, 1], Some(DType::BFloat16)).unwrap(),
        );
        assert!(w.linear("q", 32, 32).is_err());
        w.tensors.insert(
            "q.biases".into(),
            MxArray::ones(&[32, 2], Some(DType::BFloat16)).unwrap(),
        );
        assert!(w.linear("q", 32, 32).is_err());
        w.tensors.insert(
            "q.biases".into(),
            MxArray::ones(&[32, 1], Some(DType::BFloat16)).unwrap(),
        );
        assert!(w.linear("q", 32, 32).unwrap().is_quantized());
        w.has_quant_config = false;
        assert!(w.linear("q", 32, 32).is_err());
    }
    #[test]
    fn projector_aliases_preserve_quantization_sidecars() {
        assert_eq!(
            canonical_key("projector.layers.0.scales"),
            "projector.0.scales"
        );
        assert_eq!(canonical_key("linear2.weight"), "projector.2.weight");
        assert_eq!(
            canonical_key("model.layers.3.self_attn.q_proj.weight"),
            "layers.3.self_attn.q_proj.weight"
        );
    }
}
