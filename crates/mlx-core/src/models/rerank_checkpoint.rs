//! Strict checkpoint metadata, weight loading, and quantization validation
//! shared by reranker backends. Family-specific tensor names, projector layouts
//! and shape contracts stay in their adapters.

use crate::{
    array::{DType, MxArray},
    models::quant_dispatch::{self, KQuantLayout, PerLayerMode, PerLayerQuant},
    utils::safetensors::load_safetensors_lazy,
};
use napi::{Error, Result};
use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, Visitor},
};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    fmt,
    io::Read,
    marker::PhantomData,
    path::Path,
};

pub(crate) fn invalid(message: impl fmt::Display) -> Error {
    Error::from_reason(format!("rerank_invalid_model: {message}"))
}

// serde's default maps accept repeated JSON keys. Check before MLX's lazy
// reader so duplicate tensors or index entries cannot select the last value.
pub(super) struct UniqueMap<V>(pub(super) HashMap<String, V>);
impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct MapVisitor<V>(PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for MapVisitor<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut out = HashMap::new();
                while let Some((key, value)) = map.next_entry::<String, V>()? {
                    if out.insert(key.clone(), value).is_some() {
                        return Err(M::Error::custom(format!("duplicate key {key}")));
                    }
                }
                Ok(UniqueMap(out))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

pub(super) fn validate_header(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path).map_err(invalid)?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(invalid)?;
    let size = u64::from_le_bytes(size);
    // The safetensors specification's header limit, independent of model size.
    if size > 100_000_000 || size > file.metadata().map_err(invalid)?.len().saturating_sub(8) {
        return Err(invalid("invalid safetensors header length"));
    }
    let mut bytes = vec![0; size as usize];
    file.read_exact(&mut bytes).map_err(invalid)?;
    serde_json::from_slice::<UniqueMap<Value>>(&bytes).map_err(invalid)?;
    Ok(())
}

/// Strictly load single-file or indexed-sharded safetensors weights with
/// duplicate-key/alias detection, header JSON validation, and optional sidecar
/// loading (e.g. `Some(("projector.safetensors", "projector."))` for Jina).
pub(crate) fn read_weights_with<F>(
    dir: &Path,
    mut canonicalize: F,
    optional_sidecar: Option<(&str, &str)>,
) -> Result<HashMap<String, MxArray>>
where
    F: FnMut(&str) -> String,
{
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
            if !canonical.insert(canonicalize(key)) {
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

    let sidecar_in_index = optional_sidecar.is_some_and(|(file, _)| files.contains(file));
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
            if !canonical.insert(canonicalize(&key)) {
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
    if let Some((sidecar_file, required_prefix)) = optional_sidecar
        && !sidecar_in_index
        && dir.join(sidecar_file).exists()
    {
        let sidecar_path = dir.join(sidecar_file);
        validate_header(&sidecar_path)?;
        for (key, value) in load_safetensors_lazy(&sidecar_path)? {
            let normalized = canonicalize(&key);
            if !normalized.starts_with(required_prefix) || !canonical.insert(normalized) {
                return Err(invalid(format!(
                    "unexpected or conflicting projector tensor {key}"
                )));
            }
            out.insert(key, value);
        }
    }
    Ok(out)
}

/// Validate that a reranker quantization declaration is supported (4-bit affine
/// or MXFP4 with row-major layout and no input activation amax).
pub(crate) fn validate_quantization(q: PerLayerQuant, context: &str) -> Result<()> {
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

/// Validate a dense or quantized 2D weight matrix and its quantization sidecars.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_matrix_weights(
    tensors: &HashMap<String, MxArray>,
    overrides: &HashMap<String, PerLayerQuant>,
    default: PerLayerQuant,
    has_quant_config: bool,
    base: &str,
    rows: i64,
    columns: i64,
    allow_bias: bool,
) -> Result<()> {
    if !allow_bias && tensors.contains_key(&format!("{base}.bias")) {
        return Err(invalid(format!("unsupported bias tensor for {base}")));
    }
    let w = tensors
        .get(&format!("{base}.weight"))
        .ok_or_else(|| invalid(format!("missing tensor {base}.weight")))?;
    let shape = w.shape()?;
    if tensors.contains_key(&format!("{base}.scales")) {
        if !has_quant_config {
            return Err(invalid(
                "packed weights require explicit quantization metadata",
            ));
        }
        let q = quant_dispatch::effective_plq_for(base, overrides, default, None);
        validate_quantization(q, base)?;
        if columns % i64::from(q.group_size) != 0 || columns % 8 != 0 {
            return Err(invalid(format!("invalid quantization geometry for {base}")));
        }
        if shape.as_ref() != [rows, columns / 8] || w.dtype()? != DType::Uint32 {
            return Err(invalid(format!("invalid packed shape/dtype for {base}")));
        }
        let scales = tensors
            .get(&format!("{base}.scales"))
            .ok_or_else(|| invalid(format!("missing tensor {base}.scales")))?;
        if scales.shape()?.as_ref() != [rows, columns / i64::from(q.group_size)] {
            return Err(invalid(format!("invalid scales for {base}")));
        }
        if q.mode == PerLayerMode::Affine {
            let biases = tensors
                .get(&format!("{base}.biases"))
                .ok_or_else(|| invalid(format!("missing tensor {base}.biases")))?;
            if biases.shape()?.as_ref() != scales.shape()?.as_ref() {
                return Err(invalid(format!("invalid biases for {base}")));
            }
            quant_dispatch::ensure_dense_weight_floating(base, scales)?;
            quant_dispatch::ensure_dense_weight_floating(base, biases)?;
        } else if scales.dtype()? != DType::Uint8 || tensors.contains_key(&format!("{base}.biases"))
        {
            return Err(invalid(format!("invalid MXFP4 sidecars for {base}")));
        }
    } else {
        quant_dispatch::ensure_dense_weight_floating(base, w)?;
        if shape.as_ref() != [rows, columns] || tensors.contains_key(&format!("{base}.biases")) {
            return Err(invalid(format!(
                "invalid dense shape or orphan sidecar for {base}"
            )));
        }
    }
    Ok(())
}

/// Report the distinct weight storage formats present in a reranker checkpoint.
pub(crate) fn weight_formats(
    tensors: &HashMap<String, MxArray>,
    overrides: &HashMap<String, PerLayerQuant>,
    default: PerLayerQuant,
) -> Vec<&'static str> {
    let mut formats = BTreeSet::new();
    for key in tensors.keys().filter_map(|k| k.strip_suffix(".weight")) {
        if tensors.contains_key(&format!("{key}.scales")) {
            let q = quant_dispatch::effective_plq_for(key, overrides, default, None);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_header_and_index_keys_are_rejected() {
        assert!(serde_json::from_str::<UniqueMap<Value>>(r#"{"x":{},"x":{}}"#).is_err());
        assert!(serde_json::from_str::<UniqueMap<String>>(r#"{"x":"a","x":"b"}"#).is_err());
        assert!(serde_json::from_str::<UniqueMap<String>>(r#"{"x":"a","y":"b"}"#).is_ok());
    }
}
