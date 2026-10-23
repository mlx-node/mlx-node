//! Cache-owned original F32 operands. This module stores capabilities only;
//! it neither selects source arithmetic nor changes native model parameters.
//!
//! The primary import transaction publishes `.complete` after this private
//! directory. Its manifest binds stat/path source identity, never `.complete`,
//! so the marker can safely bind the manifest without a digest cycle.
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(super) const LAYOUT: &str = "tiled-moe-v2-source-operands-v1";
const PRIVATE_DIR: &str = ".mlx-source-math";
const MANIFEST: &str = "manifest.json";
const PAYLOAD: &str = "weights.safetensors";
const CONTRACT: &str = "qwen35moe-original-f32-norms-v1";
const MAPPING: &str = "gguf-qwen35moe-hf-v1";
const MAX_MANIFEST: u64 = 128 * 1024;
const MAX_PAYLOAD: u64 = 1024 * 1024;
const EXPECTED_BYTES: usize = 684_032;

type Result<T> = io::Result<T>;
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
pub(super) fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn is_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ImportIdentity {
    /// Existing importer identity: canonical path + size + inode/ctime (Unix).
    /// This is explicitly NOT a full GGUF content hash.
    pub source_stat_path_sha256: String,
    pub source_bytes: u64,
    pub assets_sha256: String,
}
impl ImportIdentity {
    fn validate(&self) -> Result<()> {
        if self.source_bytes == 0
            || !is_sha(&self.source_stat_path_sha256)
            || !is_sha(&self.assets_sha256)
        {
            return Err(invalid("invalid source operand import identity"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Geometry {
    pub architecture: String,
    pub hidden_size: u64,
    pub layers: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub full_attention_interval: u64,
    pub experts: u64,
    pub top_k: u64,
    pub expert_width: u64,
}
impl Geometry {
    pub(super) fn supported(&self) -> bool {
        self.architecture == "qwen35moe"
            && self.hidden_size == 2048
            && self.layers == 40
            && self.attention_heads == 16
            && self.kv_heads == 2
            && self.head_dim == 256
            && self.full_attention_interval == 4
            && self.experts == 256
            && self.top_k == 8
            && self.expert_width == 512
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ExpectedTensor {
    pub original: String,
    pub mapped: String,
    pub width: usize,
}
impl ExpectedTensor {
    pub(super) fn matches_original(&self, name: &str) -> bool {
        name == self.original
            || (self.original.contains(".post_attention_norm.")
                && name == self.original.replace(".post_attention_norm.", ".ffn_norm."))
    }
}

/// Explicit map, sorted by native name. Original GGUF norms contain final
/// trained values; no +1 or other transform belongs in this preservation path.
pub(super) fn expected_tensors() -> Vec<ExpectedTensor> {
    let mut result = vec![ExpectedTensor {
        original: "output_norm.weight".into(),
        mapped: "model.norm.weight".into(),
        width: 2048,
    }];
    for layer in 0..40 {
        for (original, mapped) in [
            ("attn_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
        ] {
            result.push(ExpectedTensor {
                original: format!("blk.{layer}.{original}.weight"),
                mapped: format!("model.layers.{layer}.{mapped}.weight"),
                width: 2048,
            });
        }
        if layer % 4 == 3 {
            for kind in ['q', 'k'] {
                result.push(ExpectedTensor {
                    original: format!("blk.{layer}.attn_{kind}_norm.weight"),
                    mapped: format!("model.layers.{layer}.self_attn.{kind}_norm.weight"),
                    width: 256,
                });
            }
        }
    }
    result.sort_by(|a, b| a.mapped.cmp(&b.mapped));
    result
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    original_name: String,
    mapped_name: String,
    original_dtype: String,
    original_shape: Vec<u64>,
    /// Absolute range in the source GGUF, including its data-section offset.
    original_byte_offset: u64,
    payload_offsets: [usize; 2],
    f32_sha256: String,
    native_bf16_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum Capability {
    Unavailable {
        reason: String,
    },
    Available {
        payload_file: String,
        payload_bytes: u64,
        payload_sha256: String,
        tensors: Vec<TensorRecord>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    contract: String,
    mapping: String,
    source_identity_kind: String,
    import: ImportIdentity,
    geometry: Geometry,
    native_dtype: String,
    /// Frequency operands are derived, not source GGUF tensors. A future
    /// separately versioned algorithm must prove exact source rounding.
    derived_qk_frequencies: String,
    capability: Capability,
}
#[derive(Clone)]
pub(super) struct OriginalTensor {
    pub mapped_name: String,
    pub original_name: String,
    pub original_shape: Vec<u64>,
    pub original_byte_offset: u64,
    pub f32_bytes: Vec<u8>,
}

pub(super) struct Preservation {
    manifest: Manifest,
    payload: Option<Vec<u8>>,
}
impl Preservation {
    pub(super) fn unavailable(
        import: ImportIdentity,
        geometry: Geometry,
        reason: &str,
    ) -> Result<Self> {
        import.validate()?;
        if reason.is_empty() || reason.len() > 128 {
            return Err(invalid("invalid unavailable reason"));
        }
        Ok(Self {
            manifest: Manifest {
                schema: 1,
                contract: CONTRACT.into(),
                mapping: MAPPING.into(),
                source_identity_kind: "stat-path-v1-not-content-hash".into(),
                import,
                geometry,
                native_dtype: "BF16".into(),
                derived_qk_frequencies: "not_preserved".into(),
                capability: Capability::Unavailable {
                    reason: reason.into(),
                },
            },
            payload: None,
        })
    }

    pub(super) fn capture(
        import: ImportIdentity,
        geometry: Geometry,
        tensors: Vec<OriginalTensor>,
    ) -> Result<Self> {
        let mut preservation = Self::unavailable(import, geometry, "pending")?;
        if !preservation.manifest.geometry.supported() {
            return Err(invalid("unsupported source operand geometry"));
        }
        let expected = expected_tensors();
        let mut tensors = tensors;
        tensors.sort_by(|a, b| a.mapped_name.cmp(&b.mapped_name));
        if tensors.len() != expected.len() {
            return Err(invalid("incomplete source operand set"));
        }
        let mut header = BTreeMap::new();
        let mut records = Vec::with_capacity(tensors.len());
        let mut data = Vec::with_capacity(EXPECTED_BYTES);
        for (tensor, expected) in tensors.iter().zip(&expected) {
            if !expected.matches_original(&tensor.original_name)
                || tensor.mapped_name != expected.mapped
                || tensor.original_shape != [expected.width as u64]
                || tensor.f32_bytes.len() != expected.width * 4
            {
                return Err(invalid("source operand name/shape/length mismatch"));
            }
            let end = tensor
                .original_byte_offset
                .checked_add(tensor.f32_bytes.len() as u64)
                .ok_or_else(|| invalid("GGUF operand range overflow"))?;
            if end > preservation.manifest.import.source_bytes {
                return Err(invalid("GGUF operand outside source"));
            }
            let bf16 = bf16_bytes(&tensor.f32_bytes)?;
            let offsets = [data.len(), data.len() + tensor.f32_bytes.len()];
            header.insert(tensor.mapped_name.clone(), serde_json::json!({"dtype":"F32", "shape":[expected.width], "data_offsets":offsets}));
            records.push(TensorRecord {
                original_name: tensor.original_name.clone(),
                mapped_name: tensor.mapped_name.clone(),
                original_dtype: "F32".into(),
                original_shape: tensor.original_shape.clone(),
                original_byte_offset: tensor.original_byte_offset,
                payload_offsets: offsets,
                f32_sha256: sha256(&tensor.f32_bytes),
                native_bf16_sha256: sha256(&bf16),
            });
            data.extend_from_slice(&tensor.f32_bytes);
        }
        if data.len() != EXPECTED_BYTES {
            return Err(invalid("unexpected source operand byte count"));
        }
        let mut header = serde_json::to_vec(&header).map_err(invalid_serde)?;
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut payload = Vec::with_capacity(8 + header.len() + data.len());
        payload.extend_from_slice(&(header.len() as u64).to_le_bytes());
        payload.extend_from_slice(&header);
        payload.extend_from_slice(&data);
        preservation.manifest.capability = Capability::Available {
            payload_file: PAYLOAD.into(),
            payload_bytes: payload.len() as u64,
            payload_sha256: sha256(&payload),
            tensors: records,
        };
        preservation.payload = Some(payload);
        Ok(preservation)
    }

    pub(super) fn native_digests(&self) -> Vec<(&str, &str)> {
        match &self.manifest.capability {
            Capability::Available { tensors, .. } => tensors
                .iter()
                .map(|t| (t.mapped_name.as_str(), t.native_bf16_sha256.as_str()))
                .collect(),
            Capability::Unavailable { .. } => Vec::new(),
        }
    }

    pub(super) fn publish(&self, staging: &Path) -> Result<String> {
        self.publish_with(staging, || Ok(()))
    }
    /// Private injection point tests failure after the payload write. No
    /// production environment flag or public API can inject publication faults.
    fn publish_with(
        &self,
        staging: &Path,
        before_manifest: impl FnOnce() -> Result<()>,
    ) -> Result<String> {
        let final_dir = staging.join(PRIVATE_DIR);
        if final_dir.exists() || fs::symlink_metadata(&final_dir).is_ok() {
            return Err(invalid("private source operands already exist"));
        }
        let pending = staging.join(format!(
            ".mlx-source-math.pending-{}",
            uuid::Uuid::new_v4().simple()
        ));
        fs::create_dir(&pending)?;
        let result = (|| {
            if let Some(payload) = &self.payload {
                write_new(&pending.join(PAYLOAD), payload)?;
            }
            before_manifest()?;
            let manifest = serde_json::to_vec_pretty(&self.manifest).map_err(invalid_serde)?;
            if manifest.len() as u64 > MAX_MANIFEST {
                return Err(invalid("source manifest too large"));
            }
            write_new(&pending.join(MANIFEST), &manifest)?;
            // No .complete exists yet. The outer importer owns publication of
            // the whole staging tree and rechecks source/assets before rename.
            fs::rename(&pending, &final_dir)?;
            Ok(sha256(&manifest))
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&pending);
        }
        result
    }
}
fn invalid_serde(error: serde_json::Error) -> io::Error {
    invalid(error.to_string())
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}
/// Exact IEEE BF16 round-to-nearest-even, including signed zero. Original
/// operand bytes stay unchanged; these bytes witness their ordinary narrowing.
pub(super) fn bf16_bytes(f32_bytes: &[u8]) -> Result<Vec<u8>> {
    if !f32_bytes.len().is_multiple_of(4) {
        return Err(invalid("unaligned F32 payload"));
    }
    let mut output = Vec::with_capacity(f32_bytes.len() / 2);
    for bytes in f32_bytes.as_chunks::<4>().0 {
        let bits = u32::from_le_bytes(*bytes);
        if !f32::from_bits(bits).is_finite() {
            return Err(invalid("nonfinite source norm"));
        }
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16;
        if (rounded & 0x7f80) == 0x7f80 {
            return Err(invalid("source norm overflows BF16"));
        }
        output.extend_from_slice(&(rounded as u16).to_le_bytes());
    }
    Ok(output)
}
fn bounded_read(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > maximum {
        return Err(invalid("invalid private source operand file"));
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() || bytes.len() as u64 > maximum {
        return Err(invalid("private operand file changed while reading"));
    }
    Ok(bytes)
}
fn private_directory(root: &Path) -> Result<std::path::PathBuf> {
    let dir = root.join(PRIVATE_DIR);
    if !fs::symlink_metadata(&dir)?.file_type().is_dir() {
        return Err(invalid("source operand directory must not be a symlink"));
    }
    Ok(dir)
}
/// Validate owned small bytes on a cache hit. Never follow a path supplied by
/// the manifest; filenames and tensor set are fixed by this schema.
pub(super) fn validate(
    root: &Path,
    identity: &ImportIdentity,
    expected_manifest: Option<&str>,
) -> Result<String> {
    identity.validate()?;
    let dir = private_directory(root)?;
    let bytes = bounded_read(&dir.join(MANIFEST), MAX_MANIFEST)?;
    let digest = sha256(&bytes);
    if expected_manifest.is_some_and(|expected| expected != digest) {
        return Err(invalid("source operand manifest digest mismatch"));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(invalid_serde)?;
    if manifest.schema != 1
        || manifest.contract != CONTRACT
        || manifest.mapping != MAPPING
        || manifest.source_identity_kind != "stat-path-v1-not-content-hash"
        || &manifest.import != identity
        || manifest.native_dtype != "BF16"
        || manifest.derived_qk_frequencies != "not_preserved"
    {
        return Err(invalid("source operand manifest identity/schema mismatch"));
    }
    match &manifest.capability {
        Capability::Unavailable { reason } => {
            if reason.is_empty()
                || reason.len() > 128
                || fs::symlink_metadata(dir.join(PAYLOAD)).is_ok()
            {
                return Err(invalid("invalid unavailable source capability"));
            }
        }
        Capability::Available {
            payload_file,
            payload_bytes,
            payload_sha256,
            tensors,
        } => {
            if !manifest.geometry.supported()
                || payload_file != PAYLOAD
                || *payload_bytes > MAX_PAYLOAD
            {
                return Err(invalid("unsupported available source capability"));
            }
            let payload = bounded_read(&dir.join(PAYLOAD), MAX_PAYLOAD)?;
            if payload.len() as u64 != *payload_bytes
                || sha256(&payload) != *payload_sha256
                || payload.len() < 8
            {
                return Err(invalid("source operand payload digest/length mismatch"));
            }
            let encoded: [u8; 8] = payload[..8]
                .try_into()
                .map_err(|_| invalid("invalid source operand payload header"))?;
            let header_len = usize::try_from(u64::from_le_bytes(encoded))
                .map_err(|_| invalid("payload header overflow"))?;
            let data_start = 8usize
                .checked_add(header_len)
                .filter(|n| *n <= payload.len())
                .ok_or_else(|| invalid("payload header outside file"))?;
            let header: BTreeMap<String, serde_json::Value> =
                serde_json::from_slice(&payload[8..data_start]).map_err(invalid_serde)?;
            let expected = expected_tensors();
            if tensors.len() != expected.len()
                || header.len() != expected.len()
                || payload.len() - data_start != EXPECTED_BYTES
            {
                return Err(invalid("incomplete source payload"));
            }
            let mut offset = 0;
            for (tensor, expected) in tensors.iter().zip(&expected) {
                let end = offset + expected.width * 4;
                if tensor.mapped_name != expected.mapped
                    || !expected.matches_original(&tensor.original_name)
                    || tensor.original_dtype != "F32"
                    || tensor.original_shape != [expected.width as u64]
                    || tensor.payload_offsets != [offset, end]
                    || tensor
                        .original_byte_offset
                        .checked_add((expected.width * 4) as u64)
                        .is_none_or(|end| end > identity.source_bytes)
                    || header.get(&expected.mapped)
                        != Some(
                            &serde_json::json!({"dtype":"F32", "shape":[expected.width], "data_offsets":[offset,end]}),
                        )
                {
                    return Err(invalid("source tensor descriptor mismatch"));
                }
                let values = &payload[data_start + offset..data_start + end];
                if sha256(values) != tensor.f32_sha256
                    || sha256(&bf16_bytes(values)?) != tensor.native_bf16_sha256
                {
                    return Err(invalid("source tensor digest mismatch"));
                }
                offset = end;
            }
        }
    }
    Ok(digest)
}
/// Marker parsing is exact and rejects duplicate declarations. An old native
/// directory remains native-only; it cannot satisfy this new-layout cache hit.
pub(super) fn cache_is_current(root: &Path, identity: &ImportIdentity) -> bool {
    let Ok(marker) = bounded_read(&root.join(".complete"), 32 * 1024) else {
        return false;
    };
    let Ok(marker) = std::str::from_utf8(&marker) else {
        return false;
    };
    let values = |key: &str| {
        marker
            .lines()
            .filter_map(move |line| line.strip_prefix(key))
            .collect::<Vec<_>>()
    };
    let hashes = values("source_operands_manifest_sha256=");
    values("source_operands_schema=") == ["1"]
        && values("layout=") == [LAYOUT]
        && hashes.len() == 1
        && is_sha(hashes[0])
        && validate(root, identity, Some(hashes[0])).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            let p =
                std::env::temp_dir().join(format!("mlx-source-operands-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn identity() -> ImportIdentity {
        ImportIdentity {
            source_stat_path_sha256: "a".repeat(64),
            source_bytes: 2_000_000,
            assets_sha256: "b".repeat(64),
        }
    }
    fn geometry() -> Geometry {
        Geometry {
            architecture: "qwen35moe".into(),
            hidden_size: 2048,
            layers: 40,
            attention_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            full_attention_interval: 4,
            experts: 256,
            top_k: 8,
            expert_width: 512,
        }
    }
    fn originals() -> Vec<OriginalTensor> {
        let mut offset = 128;
        expected_tensors()
            .into_iter()
            .enumerate()
            .map(|(index, expected)| {
                let values = (0..expected.width)
                    .flat_map(|n| (1.0 + ((index * 31 + n) % 127) as f32 * 0.00013).to_le_bytes())
                    .collect::<Vec<_>>();
                let tensor = OriginalTensor {
                    mapped_name: expected.mapped,
                    original_name: expected.original,
                    original_shape: vec![expected.width as u64],
                    original_byte_offset: offset,
                    f32_bytes: values,
                };
                offset += tensor.f32_bytes.len() as u64;
                tensor
            })
            .collect()
    }
    fn available() -> Preservation {
        Preservation::capture(identity(), geometry(), originals()).unwrap()
    }
    fn marker(root: &Path, hash: &str) {
        fs::write(root.join(".complete"), format!("layout={LAYOUT}\nsource_operands_schema=1\nsource_operands_manifest_sha256={hash}\n")).unwrap();
    }
    #[test]
    fn original_f32_roundtrip_exact_and_native_counterparts_distinct() {
        let temp = Temp::new();
        let preserved = available();
        let hash = preserved.publish(&temp.0).unwrap();
        assert_eq!(expected_tensors().len(), 101);
        assert_eq!(
            originals().iter().map(|t| t.f32_bytes.len()).sum::<usize>(),
            EXPECTED_BYTES
        );
        assert_eq!(validate(&temp.0, &identity(), Some(&hash)).unwrap(), hash);
        let payload = fs::read(temp.0.join(PRIVATE_DIR).join(PAYLOAD)).unwrap();
        let start = 8 + u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
        assert_eq!(
            &payload[start..],
            originals()
                .iter()
                .flat_map(|t| t.f32_bytes.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(preserved.native_digests().len(), 101);
        assert!(
            preserved
                .native_digests()
                .iter()
                .zip(originals())
                .all(|((_, digest), t)| *digest == sha256(&bf16_bytes(&t.f32_bytes).unwrap()))
        );
        assert!(
            !String::from_utf8(fs::read(temp.0.join(PRIVATE_DIR).join(MANIFEST)).unwrap())
                .unwrap()
                .contains(".complete")
        );
        assert!(!temp.0.join("weights.safetensors").exists());
    }
    #[test]
    fn bf16_rne_ties_signed_zero_and_nonfinite() {
        let values = [0u32, 0x80000000, 0x3f808000, 0x3f818000, 0x3f800001];
        let bytes = values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            bf16_bytes(&bytes).unwrap(),
            [0u16, 0x8000, 0x3f80, 0x3f82, 0x3f80]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        );
        for bits in [
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            f32::MAX.to_bits(),
        ] {
            assert!(bf16_bytes(&bits.to_le_bytes()).is_err());
        }
    }
    #[test]
    fn malformed_partial_duplicate_and_changed_geometry_refused() {
        let mut aliases = originals();
        for tensor in &mut aliases {
            tensor.original_name = tensor
                .original_name
                .replace(".post_attention_norm.", ".ffn_norm.");
        }
        let temp = Temp::new();
        let alias = Preservation::capture(identity(), geometry(), aliases).unwrap();
        let hash = alias.publish(&temp.0).unwrap();
        validate(&temp.0, &identity(), Some(&hash)).unwrap();
        let mut values = originals();
        values.pop();
        assert!(Preservation::capture(identity(), geometry(), values).is_err());
        let mut values = originals();
        values[1] = values[0].clone();
        assert!(Preservation::capture(identity(), geometry(), values).is_err());
        let mut values = originals();
        values[0].original_shape = vec![1, 2048];
        assert!(Preservation::capture(identity(), geometry(), values).is_err());
        let mut values = originals();
        values[0].original_byte_offset = u64::MAX;
        assert!(Preservation::capture(identity(), geometry(), values).is_err());
        let mut values = originals();
        values[0].f32_bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Preservation::capture(identity(), geometry(), values).is_err());
        let mut g = geometry();
        g.architecture = "qwen35".into();
        assert!(Preservation::capture(identity(), g, originals()).is_err());
    }
    #[test]
    fn private_publish_failure_is_atomic_and_never_mutates_main_weights() {
        let temp = Temp::new();
        let native = b"existing main safetensors bytes";
        fs::write(temp.0.join("model.safetensors"), native).unwrap();
        assert!(
            available()
                .publish_with(&temp.0, || Err(io::Error::other("injected after payload")))
                .is_err()
        );
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 1);
        let hash = available().publish(&temp.0).unwrap();
        assert_eq!(fs::read(temp.0.join("model.safetensors")).unwrap(), native);
        assert!(!cache_is_current(&temp.0, &identity()));
        marker(&temp.0, &hash);
        assert!(cache_is_current(&temp.0, &identity()));
        assert!(available().publish(&temp.0).is_err());
        assert_eq!(validate(&temp.0, &identity(), Some(&hash)).unwrap(), hash);
    }
    #[test]
    fn corrupt_missing_payload_identity_and_duplicate_marker_reject_hits() {
        let temp = Temp::new();
        let hash = available().publish(&temp.0).unwrap();
        marker(&temp.0, &hash);
        let mut changed = identity();
        changed.source_stat_path_sha256 = "c".repeat(64);
        assert!(!cache_is_current(&temp.0, &changed));
        let p = temp.0.join(PRIVATE_DIR).join(PAYLOAD);
        let original = fs::read(&p).unwrap();
        let mut corrupt = original.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(&p, corrupt).unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
        fs::remove_file(&p).unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
        fs::write(&p, original).unwrap();
        assert!(cache_is_current(&temp.0, &identity()));
        let mut f = OpenOptions::new()
            .append(true)
            .open(temp.0.join(".complete"))
            .unwrap();
        writeln!(f, "source_operands_schema=1").unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
    }
    #[test]
    fn unavailable_is_explicit_and_legacy_is_native_only() {
        let temp = Temp::new();
        fs::write(temp.0.join(".complete"), "layout=tiled-moe-v1\n").unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
        let mut g = geometry();
        g.hidden_size = 16;
        let unavailable = Preservation::unavailable(identity(), g, "unsupported_geometry").unwrap();
        let hash = unavailable.publish(&temp.0).unwrap();
        marker(&temp.0, &hash);
        assert!(cache_is_current(&temp.0, &identity()));
        assert!(unavailable.native_digests().is_empty());
        assert!(!temp.0.join(PRIVATE_DIR).join(PAYLOAD).exists());
        fs::write(temp.0.join(PRIVATE_DIR).join(PAYLOAD), b"unexpected").unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
    }
    #[test]
    fn untrusted_paths_and_descriptor_changes_are_rejected_even_with_rehashed_marker() {
        let temp = Temp::new();
        available().publish(&temp.0).unwrap();
        let path = temp.0.join(PRIVATE_DIR).join(MANIFEST);
        let original = fs::read(&path).unwrap();
        for (field, value) in [
            ("payload_file", serde_json::json!("../model.safetensors")),
            ("payload_bytes", serde_json::json!(u64::MAX)),
        ] {
            let mut m: serde_json::Value = serde_json::from_slice(&original).unwrap();
            m["capability"][field] = value;
            let bytes = serde_json::to_vec(&m).unwrap();
            fs::write(&path, &bytes).unwrap();
            marker(&temp.0, &sha256(&bytes));
            assert!(!cache_is_current(&temp.0, &identity()));
        }
        let mut m: serde_json::Value = serde_json::from_slice(&original).unwrap();
        m["capability"]["tensors"][0]["original_shape"] = serde_json::json!([4096]);
        let bytes = serde_json::to_vec(&m).unwrap();
        fs::write(path, &bytes).unwrap();
        marker(&temp.0, &sha256(&bytes));
        assert!(!cache_is_current(&temp.0, &identity()));
    }
    #[cfg(unix)]
    #[test]
    fn symlink_payload_and_directory_are_not_capabilities() {
        let temp = Temp::new();
        let outside = Temp::new();
        let hash = available().publish(&temp.0).unwrap();
        marker(&temp.0, &hash);
        let p = temp.0.join(PRIVATE_DIR).join(PAYLOAD);
        fs::rename(&p, outside.0.join(PAYLOAD)).unwrap();
        std::os::unix::fs::symlink(outside.0.join(PAYLOAD), &p).unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
        fs::remove_file(&p).unwrap();
        fs::rename(outside.0.join(PAYLOAD), &p).unwrap();
        assert!(cache_is_current(&temp.0, &identity()));
        fs::rename(temp.0.join(PRIVATE_DIR), outside.0.join(PRIVATE_DIR)).unwrap();
        std::os::unix::fs::symlink(outside.0.join(PRIVATE_DIR), temp.0.join(PRIVATE_DIR)).unwrap();
        assert!(!cache_is_current(&temp.0, &identity()));
    }
}
