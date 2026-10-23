//! Load-only original Q/K norm and frequency diagnostic for the measured GGUF.
//! Native norm parameters stay unchanged; only eligible T8 attention uses these
//! separately owned F32 operands. The immutable model policy blocks persistence,
//! training and export while this alternate arithmetic identity is active.
use super::{
    config::Qwen3_5MoeConfig,
    decoder_layer::{AttentionType, DecoderLayer},
};
use crate::array::{DType, MxArray};
use napi::bindgen_prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

const PAYLOAD: &str = "qk-norms-rope-f32.bin";
const PAYLOAD_SHA: &str = "09cfd35ac75cf7db0603ac79573d3004b63a43fa8e081e67f163e4997b73556f";
const COMPLETE_SHA: &str = "a6b87df3e5e2451723a0d576dbdbf317a03fcb6cdf920a46487ac83b46ebde90";
const SOURCE_SHA: &str = "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f";
const PAYLOAD_BYTES: usize = 20 * 256 * 4 + 32 * 4;

pub(crate) struct SourceQkRequest {
    manifest: PathBuf,
}
pub(crate) struct SourceQkWeights {
    norms: Vec<(usize, MxArray, MxArray)>,
    frequencies: MxArray,
}
fn error(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Qwen3.5 MoE source QK diagnostic: {message}"))
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
    if file.metadata().map_err(error)?.len() > maximum as u64 {
        return Err(error("file exceeds size limit"));
    }
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(error)?;
    if bytes.len() > maximum {
        return Err(error("file grew beyond size limit"));
    }
    Ok(bytes)
}
fn values(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(error("incomplete F32 value"));
    }
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(error("nonfinite operand"));
    }
    Ok(v)
}
fn validate_rounding(source: &[f32], native: &[u16]) -> Result<()> {
    if source.len() != 256
        || native.len() != 256
        || source
            .iter()
            .zip(native)
            .any(|(&a, &b)| !a.is_finite() || half::bf16::from_f32(a).to_bits() != b)
    {
        return Err(error(
            "source norm does not round to the native BF16 parameter",
        ));
    }
    Ok(())
}
impl SourceQkRequest {
    pub(crate) fn capture() -> Result<Option<Self>> {
        Self::from_value(std::env::var_os("MLX_QWEN35_MOE_SOURCE_QK"))
    }
    fn from_value(value: Option<std::ffi::OsString>) -> Result<Option<Self>> {
        let Some(value) = value.filter(|x| x != "0") else {
            return Ok(None);
        };
        if value.is_empty() {
            return Err(error("manifest path is empty"));
        }
        Ok(Some(Self {
            manifest: value.into(),
        }))
    }
    pub(crate) fn validate_profile(config: &Qwen3_5MoeConfig, has_vision: bool) -> Result<()> {
        if has_vision
            || config.hidden_size != 2048
            || config.num_layers != 40
            || config.num_experts != 256
            || config.num_experts_per_tok != 8
            || config.num_heads != 16
            || config.num_kv_heads != 2
            || config.head_dim != 256
            || config.full_attention_interval != 4
            || config.rope_dims() != 64
            || (config.rope_theta as f32).to_bits() != 10_000_000_f32.to_bits()
            || (config.rms_norm_eps as f32).to_bits() != 1e-6_f32.to_bits()
        {
            return Err(error(
                "requires measured text-only H2048/L40/Q16/KV2/D256/Rot64/1e7 profile",
            ));
        }
        Ok(())
    }
    pub(crate) fn load(
        &self,
        model_path: &Path,
        config: &Qwen3_5MoeConfig,
        has_vision: bool,
        params: &HashMap<String, MxArray>,
    ) -> Result<SourceQkWeights> {
        Self::validate_profile(config, has_vision)?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&read_bounded(&self.manifest, 64 * 1024)?).map_err(error)?;
        if manifest["source_sha256"] != SOURCE_SHA
            || manifest["source_bytes"] != 21_713_463_040_u64
            || manifest["source_revision"] != "12393612fd4f730ff5aadc23e9b8f9648aa49ceb"
            || manifest["prepared_complete_sha256"] != COMPLETE_SHA
            || manifest["payload"]["sha256"] != PAYLOAD_SHA
            || manifest["payload"]["bytes"].as_u64() != Some(PAYLOAD_BYTES as u64)
        {
            return Err(error(
                "manifest identity differs from verified source export",
            ));
        }
        let payload = read_bounded(
            &self
                .manifest
                .parent()
                .ok_or_else(|| error("manifest parent missing"))?
                .join(PAYLOAD),
            PAYLOAD_BYTES,
        )?;
        if payload.len() != PAYLOAD_BYTES || digest(&payload) != PAYLOAD_SHA {
            return Err(error("payload bytes differ from verified source export"));
        }
        // Follow the same prepared shard symlink as the model loader, then
        // validate the adjacent immutable import marker and each norm value.
        let shard = std::fs::canonicalize(model_path.join("model.safetensors")).map_err(error)?;
        if shard.file_name().is_none_or(|x| x != "model.safetensors") {
            return Err(error("unexpected prepared shard name"));
        }
        let complete = read_bounded(
            &shard
                .parent()
                .ok_or_else(|| error("shard parent missing"))?
                .join(".complete"),
            64 * 1024,
        )?;
        if digest(&complete) != COMPLETE_SHA {
            return Err(error("prepared cache identity differs"));
        }
        let decoded = values(&payload)?;
        let mut norms = Vec::with_capacity(10);
        for (index, layer) in (3..40).step_by(4).enumerate() {
            let mut pair = Vec::with_capacity(2);
            for (side, name) in ["q", "k"].into_iter().enumerate() {
                let key = format!("layers.{layer}.self_attn.{name}_norm.weight");
                let native = params
                    .get(&key)
                    .ok_or_else(|| error(format!("missing {key}")))?;
                if native.shape()?.as_ref() != [256] || native.dtype()? != DType::BFloat16 {
                    return Err(error(format!("{key} must be BF16/[256]")));
                }
                let start = (index * 2 + side) * 256;
                let source = &decoded[start..start + 256];
                validate_rounding(source, &native.to_uint16_native()?)?;
                pair.push(MxArray::from_float32(source, &[256])?);
            }
            let k = pair.pop().unwrap();
            let q = pair.pop().unwrap();
            norms.push((layer, q, k));
        }
        Ok(SourceQkWeights {
            norms,
            frequencies: MxArray::from_float32(&decoded[5120..], &[32])?,
        })
    }
}
impl SourceQkWeights {
    pub(crate) fn install(self, layers: &mut [DecoderLayer]) -> Result<()> {
        if layers.len() != 40
            || self
                .norms
                .iter()
                .any(|(index, _, _)| !matches!(layers[*index].attn, AttentionType::Full(_)))
        {
            return Err(error("loaded attention layer inventory differs"));
        }
        // Validate every setter before mutating any layer; all arrays are
        // allocated and value-checked already. The following installs cannot
        // fail and do not replace native weights used by unsupported shapes.
        for (index, q, k) in &self.norms {
            let AttentionType::Full(attn) = &layers[*index].attn else {
                unreachable!()
            };
            attn.validate_source_qk_weights(q, k, &self.frequencies)?;
        }
        for (index, q, k) in self.norms {
            let AttentionType::Full(attn) = &mut layers[index].attn else {
                unreachable!()
            };
            attn.install_source_qk_weights(q, k, self.frequencies.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_qk_request_and_rounding_are_fail_closed() {
        assert!(SourceQkRequest::from_value(None).unwrap().is_none());
        assert!(
            SourceQkRequest::from_value(Some("0".into()))
                .unwrap()
                .is_none()
        );
        assert!(SourceQkRequest::from_value(Some("".into())).is_err());
        let request = SourceQkRequest::from_value(Some("fixture.json".into()))
            .unwrap()
            .unwrap();
        assert_eq!(request.manifest, Path::new("fixture.json"));
        let source: Vec<f32> = (0..256).map(|i| 1.0 + i as f32 * 0.00031).collect();
        let native: Vec<u16> = source
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();
        validate_rounding(&source, &native).unwrap();
        assert!(validate_rounding(&source[..255], &native).is_err());
        let mut bad = native.clone();
        bad[7] ^= 1;
        assert!(validate_rounding(&source, &bad).is_err());
        let mut bad = source;
        bad[11] = f32::NAN;
        assert!(validate_rounding(&bad, &native).is_err());
        assert!(values(&[0, 1, 2]).is_err());
        assert!(values(&f32::INFINITY.to_le_bytes()).is_err());
    }

    #[test]
    fn source_qk_rejects_changed_rotary_or_attention_identity() {
        let config: Qwen3_5MoeConfig = serde_json::from_value(serde_json::json!({
            "vocab_size":248320,"hidden_size":2048,"num_layers":40,"num_heads":16,
            "num_kv_heads":2,"intermediate_size":512,"rms_norm_eps":1e-6,"head_dim":256,
            "tie_word_embeddings":false,"max_position_embeddings":32768,"pad_token_id":0,
            "eos_token_id":1,"bos_token_id":2,"num_experts":256,"num_experts_per_tok":8,
            "rope_theta":10000000.0,"partial_rotary_factor":0.25
        }))
        .unwrap();
        SourceQkRequest::validate_profile(&config, false).unwrap();
        assert!(SourceQkRequest::validate_profile(&config, true).is_err());
        for field in 0..8 {
            let mut changed = config.clone();
            match field {
                0 => changed.head_dim = 128,
                1 => changed.num_heads = 8,
                2 => changed.num_kv_heads = 4,
                3 => changed.rope_theta = 100000.0,
                4 => changed.rms_norm_eps = 1e-5,
                5 => changed.partial_rotary_factor = 0.5,
                6 => changed.full_attention_interval = 2,
                _ => changed.num_layers = 41,
            }
            assert!(SourceQkRequest::validate_profile(&changed, false).is_err());
        }
        let mut rounded = config;
        rounded.rms_norm_eps = f64::from(1e-6_f32);
        SourceQkRequest::validate_profile(&rounded, false).unwrap();
        let policy =
            super::super::source_math_policy::SourceMoePolicy::default().with_qk_requested(true);
        assert!(policy.active() && policy.qk);
        assert!(!policy.body && !policy.routing && !policy.residual && !policy.mixer);
    }
}
