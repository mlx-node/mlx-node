//! Exact-checkpoint, load-only source residual-norm precision diagnostic.
//! No importer/cache-format changes; every sidecar byte is owned and validated
//! before replacing parameters. Activations remain BF16 through RMSNorm policy.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::Qwen3_5MoeConfig;
use crate::array::{DType, MxArray};

const WIDTH: usize = 2048;
const COUNT: usize = 81;
const PAYLOAD_NAME: &str = "residual-norms-f32.safetensors";
const PAYLOAD_SHA: &str = "fd94ba840e9d61068d5c7a612e0687647f72605bc911782a07631fb8932eeb53";
const SOURCE_SHA: &str = "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f";
const SOURCE_REVISION: &str = "12393612fd4f730ff5aadc23e9b8f9648aa49ceb";
const COMPLETE_SHA: &str = "a6b87df3e5e2451723a0d576dbdbf317a03fcb6cdf920a46487ac83b46ebde90";
const CACHE_ID: &str = "acb52f5fcc4cde1f4dc0a055d658979d84712d4d763acb642c2ac4585d06d9ac";
const SOURCE_BYTES: u64 = 21_713_463_040;

#[derive(Clone, Debug)]
pub(crate) struct SourceNormIdentity {
    pub(crate) source_reduction: bool,
    pub(crate) payload_sha256: String,
    pub(crate) prepared_complete_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceNormRequest {
    manifest: PathBuf,
    source_reduction: bool,
}
impl SourceNormRequest {
    pub(crate) fn capture() -> Result<Option<Self>> {
        let path = std::env::var_os("MLX_QWEN35_MOE_SOURCE_NORMS");
        let math = std::env::var_os("MLX_QWEN35_MOE_SOURCE_NORM_MATH");
        Self::from_values(path, math)
    }

    fn from_values(
        path: Option<std::ffi::OsString>,
        math: Option<std::ffi::OsString>,
    ) -> Result<Option<Self>> {
        let Some(path) = path.filter(|value| value != "0") else {
            return Ok(None);
        };
        if path.is_empty() {
            return Err(error("manifest path is empty"));
        }
        let source_reduction = match math.as_deref() {
            None => false,
            Some(value) if value == "reference" => false,
            Some(value) if value == "splash" => true,
            _ => return Err(error("SOURCE_NORM_MATH must be reference or splash")),
        };
        Ok(Some(Self {
            manifest: PathBuf::from(path),
            source_reduction,
        }))
    }

    pub(crate) fn replace_parameters(
        &self,
        model_path: &Path,
        config: &Qwen3_5MoeConfig,
        has_vision: bool,
        params: &mut HashMap<String, MxArray>,
    ) -> Result<SourceNormIdentity> {
        if has_vision
            || config.hidden_size != WIDTH as i32
            || config.num_layers != 40
            || config.num_experts != 256
            || config.num_experts_per_tok != 8
            || config
                .moe_intermediate_size
                .unwrap_or(config.intermediate_size)
                != 512
            || !source_epsilon_matches(config.rms_norm_eps)
        {
            return Err(error(
                "diagnostic requires the measured text-only H2048/L40/E256/top8/I512 profile",
            ));
        }
        let manifest_bytes = read_bounded(&self.manifest, 128 * 1024)?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| error(format!("invalid manifest: {e}")))?;
        validate_manifest(&manifest)?;
        let root = self
            .manifest
            .parent()
            .ok_or_else(|| error("manifest has no parent"))?;
        // Fixed basename, never the arbitrary path strings in manifest metadata.
        let payload = read_bounded(&root.join(PAYLOAD_NAME), 1024 * 1024)?;
        if sha256(&payload) != PAYLOAD_SHA {
            return Err(error(
                "payload SHA256 does not match the verified original source export",
            ));
        }
        let shard = std::fs::canonicalize(model_path.join("model.safetensors"))
            .map_err(|e| error(format!("cannot resolve prepared weight shard: {e}")))?;
        if shard
            .file_name()
            .is_none_or(|name| name != "model.safetensors")
        {
            return Err(error(
                "prepared cache must use the verified single model.safetensors shard",
            ));
        }
        let parent = shard
            .parent()
            .ok_or_else(|| error("prepared shard has no parent"))?;
        let complete = read_bounded(&parent.join(".complete"), 64 * 1024)?;
        validate_complete(&complete)?;
        let decoded = decode_payload(&payload, &manifest)?;
        let mut replacements = Vec::with_capacity(COUNT);
        for (name, values) in decoded {
            let existing = params
                .get(&name)
                .ok_or_else(|| error(format!("missing native norm {name}")))?;
            if existing.dtype()? != DType::BFloat16 || existing.shape()?.as_ref() != [WIDTH as i64]
            {
                return Err(error(format!("native norm {name} must be BF16/[2048]")));
            }
            validate_native_rounding(&name, &values, &existing.to_uint16_native()?)?;
            replacements.push((name, MxArray::from_float32(&values, &[WIDTH as i64])?));
        }
        // All structural, identity, value and allocation checks precede mutation.
        for (name, array) in replacements {
            params.insert(name, array);
        }
        Ok(SourceNormIdentity {
            source_reduction: self.source_reduction,
            payload_sha256: PAYLOAD_SHA.into(),
            prepared_complete_sha256: COMPLETE_SHA.into(),
        })
    }
}

// GGUF stores this metadata in F32; JSON may preserve its promoted F64 value.
// The actual norm kernels receive F32, so compare that exact representation.
fn source_epsilon_matches(value: f64) -> bool {
    (value as f32).to_bits() == 1e-6_f32.to_bits()
}

fn error(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Qwen3.5 MoE source norm diagnostic: {message}"))
}
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file =
        File::open(path).map_err(|e| error(format!("cannot read {}: {e}", path.display())))?;
    if file.metadata().map_err(error)?.len() > maximum {
        return Err(error("sidecar file exceeds size bound"));
    }
    let mut owned = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut owned)
        .map_err(error)?;
    if owned.len() as u64 > maximum {
        return Err(error("sidecar file grew beyond size bound"));
    }
    Ok(owned)
}
fn expected_names() -> BTreeSet<String> {
    let mut names = BTreeSet::from(["final_norm.weight".to_string()]);
    for layer in 0..40 {
        names.insert(format!("layers.{layer}.input_layernorm.weight"));
        names.insert(format!("layers.{layer}.post_attention_layernorm.weight"));
    }
    names
}
fn validate_manifest(manifest: &Value) -> Result<()> {
    for (value, expected, name) in [
        (&manifest["source"]["sha256"], SOURCE_SHA, "source SHA"),
        (
            &manifest["source"]["revision"],
            SOURCE_REVISION,
            "source revision",
        ),
        (
            &manifest["prepared_complete_sha256"],
            COMPLETE_SHA,
            "prepared marker SHA",
        ),
        (&manifest["sidecar"]["sha256"], PAYLOAD_SHA, "payload SHA"),
        (
            &manifest["prepared_complete_identity"]["source_identity_sha256"],
            CACHE_ID,
            "prepared source identity",
        ),
        (
            &manifest["prepared_complete_identity"]["format"],
            "7",
            "prepared format",
        ),
        (
            &manifest["prepared_complete_identity"]["dtype"],
            "bf16",
            "prepared dtype",
        ),
        (
            &manifest["prepared_complete_identity"]["layout"],
            "tiled-moe-v1",
            "prepared layout",
        ),
        (
            &manifest["namespace"],
            "post-sanitize Qwen3_5MoeInner parameters",
            "tensor namespace",
        ),
        (
            &manifest["value_transform"],
            "none; byte-for-byte source F32, no +1 transform",
            "value transform",
        ),
    ] {
        if value.as_str() != Some(expected) {
            return Err(error(format!("unexpected {name}")));
        }
    }
    if manifest["source"]["bytes"].as_u64() != Some(SOURCE_BYTES)
        || manifest["tensor_count"].as_u64() != Some(COUNT as u64)
        || manifest["payload_bytes"].as_u64() != Some((COUNT * WIDTH * 4) as u64)
        || manifest["sidecar"]["bytes"].as_u64() != Some(672_072)
    {
        return Err(error("unexpected source or sidecar size"));
    }
    Ok(())
}
fn validate_complete(complete: &[u8]) -> Result<()> {
    if sha256(complete) != COMPLETE_SHA {
        return Err(error(
            "prepared .complete identity differs from the source export",
        ));
    }
    Ok(())
}
fn validate_native_rounding(name: &str, values: &[f32], native: &[u16]) -> Result<()> {
    if values.len() != WIDTH || native.len() != WIDTH {
        return Err(error(format!("invalid norm length: {name}")));
    }
    for (i, (&value, &got)) in values.iter().zip(native).enumerate() {
        if !value.is_finite() || half::bf16::from_f32(value).to_bits() != got {
            return Err(error(format!(
                "source/native BF16 rounding mismatch at {name}[{i}]"
            )));
        }
    }
    Ok(())
}
fn decode_payload(bytes: &[u8], manifest: &Value) -> Result<Vec<(String, Vec<f32>)>> {
    let encoded: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| error("truncated SafeTensors length"))?
        .try_into()
        .map_err(|_| error("invalid SafeTensors length"))?;
    let header_len = usize::try_from(u64::from_le_bytes(encoded)).map_err(error)?;
    if header_len > 128 * 1024 {
        return Err(error("SafeTensors header exceeds size bound"));
    }
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| error("SafeTensors header overflow"))?;
    let header: Value = serde_json::from_slice(
        bytes
            .get(8..data_start)
            .ok_or_else(|| error("truncated SafeTensors header"))?,
    )
    .map_err(error)?;
    let header = header
        .as_object()
        .ok_or_else(|| error("SafeTensors header is not a map"))?;
    let names: BTreeSet<_> = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    if names != expected_names() {
        return Err(error(
            "SafeTensors norm inventory differs from exact81-name set",
        ));
    }
    let records = manifest["tensors"]
        .as_array()
        .ok_or_else(|| error("manifest tensors missing"))?;
    let mut hashes = HashMap::new();
    for record in records {
        let name = record["name"]
            .as_str()
            .ok_or_else(|| error("tensor name missing"))?;
        if record["dtype"] != "F32"
            || record["shape"] != serde_json::json!([WIDTH])
            || record["bytes"].as_u64() != Some((WIDTH * 4) as u64)
        {
            return Err(error(format!("invalid manifest tensor descriptor: {name}")));
        }
        let digest = record["sha256"]
            .as_str()
            .ok_or_else(|| error("tensor SHA missing"))?;
        if hashes.insert(name.to_string(), digest).is_some() {
            return Err(error("duplicate manifest tensor"));
        }
    }
    if hashes.keys().cloned().collect::<BTreeSet<_>>() != names {
        return Err(error("manifest norm inventory differs"));
    }
    let payload = bytes
        .get(data_start..)
        .ok_or_else(|| error("missing tensor payload"))?;
    if payload.len() != COUNT * WIDTH * 4 {
        return Err(error("unexpected tensor payload size"));
    }
    let mut decoded = Vec::with_capacity(COUNT);
    let mut ranges = Vec::with_capacity(COUNT);
    for name in names {
        let descriptor = &header[&name];
        if descriptor["dtype"] != "F32" || descriptor["shape"] != serde_json::json!([WIDTH]) {
            return Err(error(format!("invalid SafeTensors descriptor: {name}")));
        }
        let offsets = descriptor["data_offsets"]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or_else(|| error("invalid tensor offsets"))?;
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| error("invalid tensor start"))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| error("invalid tensor end"))?;
        if end.checked_sub(start) != Some(WIDTH * 4) {
            return Err(error("tensor extent mismatch"));
        }
        let data = payload
            .get(start..end)
            .ok_or_else(|| error("tensor outside payload"))?;
        if sha256(data) != hashes[&name] {
            return Err(error(format!("tensor digest differs: {name}")));
        }
        let values: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(error("nonfinite source norm"));
        }
        ranges.push((start, end));
        decoded.push((name, values));
    }
    ranges.sort_unstable();
    let mut next = 0;
    for (start, end) in ranges {
        if start != next {
            return Err(error("overlapping or incomplete tensor ranges"));
        }
        next = end;
    }
    if next != payload.len() {
        return Err(error("unreferenced tensor bytes"));
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<u8>, Value) {
        let mut header = serde_json::Map::new();
        let mut records = Vec::new();
        let mut payload = Vec::<u8>::new();
        for name in expected_names() {
            let data: Vec<u8> = (0..WIDTH)
                .flat_map(|i| (1.0 + (i % 17) as f32 * 0.00031).to_le_bytes())
                .collect();
            let start = payload.len();
            payload.extend_from_slice(&data);
            header.insert(name.clone(), serde_json::json!({"dtype":"F32","shape":[WIDTH],"data_offsets":[start,payload.len()]}));
            records.push(serde_json::json!({"name":name,"dtype":"F32","shape":[WIDTH],"bytes":WIDTH*4,"sha256":sha256(&data)}));
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.extend(payload);
        (bytes, serde_json::json!({"tensors":records}))
    }
    fn with_header(bytes: &[u8], change: impl FnOnce(&mut Value)) -> Vec<u8> {
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut header: Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
        change(&mut header);
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut result = (encoded.len() as u64).to_le_bytes().to_vec();
        result.extend(encoded);
        result.extend(&bytes[8 + length..]);
        result
    }
    #[test]
    fn source_norm_imported_epsilon_matches_actual_kernel_bits() {
        assert!(source_epsilon_matches(1e-6_f64));
        assert!(source_epsilon_matches(f64::from(1e-6_f32)));
        assert!(source_epsilon_matches(9.999999974752427e-7));
        for value in [
            0.0,
            -1e-6,
            1e-5,
            f64::NAN,
            f64::INFINITY,
            f64::from(f32::from_bits(1e-6_f32.to_bits() - 1)),
            f64::from(f32::from_bits(1e-6_f32.to_bits() + 1)),
        ] {
            assert!(!source_epsilon_matches(value), "{value:?}");
        }
    }

    #[test]
    fn source_norm_request_is_explicit_load_only_and_rejects_unknown_math() {
        assert!(
            SourceNormRequest::from_values(None, Some("unknown".into()))
                .unwrap()
                .is_none()
        );
        assert!(
            SourceNormRequest::from_values(Some("0".into()), None)
                .unwrap()
                .is_none()
        );
        assert!(SourceNormRequest::from_values(Some("".into()), None).is_err());
        assert!(
            SourceNormRequest::from_values(Some("manifest.json".into()), Some("unknown".into()))
                .is_err()
        );
        let captured =
            SourceNormRequest::from_values(Some("manifest.json".into()), Some("reference".into()))
                .unwrap()
                .unwrap();
        let source =
            SourceNormRequest::from_values(Some("manifest.json".into()), Some("splash".into()))
                .unwrap()
                .unwrap();
        assert!(source.source_reduction);
        assert!(!captured.source_reduction);
        let _other = SourceNormRequest::from_values(None, None).unwrap();
        assert_eq!(captured.manifest, Path::new("manifest.json"));
    }
    #[test]
    fn source_norm_payload_checks_names_offsets_hashes_and_finite_f32() {
        let (bytes, manifest) = fixture();
        let valid = decode_payload(&bytes, &manifest).unwrap();
        assert_eq!(valid.len(), COUNT);
        let mut duplicate = manifest.clone();
        let first = duplicate["tensors"][0].clone();
        duplicate["tensors"].as_array_mut().unwrap().push(first);
        assert!(decode_payload(&bytes, &duplicate).is_err());
        let mut missing = manifest.clone();
        missing["tensors"].as_array_mut().unwrap().pop();
        assert!(decode_payload(&bytes, &missing).is_err());
        let bad = with_header(&bytes, |h| {
            h["final_norm.weight"]["shape"] = serde_json::json!([2047]);
        });
        assert!(decode_payload(&bad, &manifest).is_err());
        let bad = with_header(&bytes, |h| {
            h["layers.0.input_layernorm.weight"]["data_offsets"] =
                serde_json::json!([0, WIDTH * 4]);
        });
        assert!(decode_payload(&bad, &manifest).is_err());
        let mut bad = bytes.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(decode_payload(&bad, &manifest).is_err());
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut bad = bytes.clone();
        bad[8 + length..12 + length].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut nonfinite = manifest.clone();
        nonfinite["tensors"][0]["sha256"] = sha256(&bad[8 + length..8 + length + WIDTH * 4]).into();
        assert!(decode_payload(&bad, &nonfinite).is_err());
    }
    #[test]
    fn source_norm_rounding_and_identity_fail_closed() {
        let values: Vec<f32> = (0..WIDTH)
            .map(|i| 1.0 + (i % 31) as f32 * 0.00031)
            .collect();
        let native: Vec<u16> = values
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        validate_native_rounding("norm", &values, &native).unwrap();
        let shifted: Vec<f32> = values.iter().map(|v| v + 1.0).collect();
        assert!(validate_native_rounding("norm", &shifted, &native).is_err());
        assert!(validate_manifest(&serde_json::json!({})).is_err());
        assert!(validate_complete(b"format=7\ndtype=bf16\n").is_err());
        assert_ne!(sha256(b"wrong source norm"), PAYLOAD_SHA);
    }
}
