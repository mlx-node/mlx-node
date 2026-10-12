use crate::{
    array::MxArray,
    engine::persistence::load_all_safetensors,
    nn::{LayerNorm, Linear, RMSNorm},
};
use napi::{Error, Result};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default = "affine")]
    pub mode: String,
}
fn affine() -> String {
    "affine".into()
}

/// Layout is declared by conversion provenance, never inferred from dimensions.
pub struct Weights {
    pub tensors: HashMap<String, MxArray>,
    pub mlx_layout: bool,
    quant: Option<Quantization>,
}
impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg: serde_json::Value = super::config::read_json(&path.join("config.json"))?;
        let format = cfg.get("mlx_node_tts_format").and_then(|x| x.as_u64());
        if cfg.get("mlx_node_tts_format").is_some() && format != Some(1) {
            return Err(Error::from_reason(
                "Unsupported TTS checkpoint format version",
            ));
        }
        let quant: Option<Quantization> = cfg
            .get("quantization")
            .filter(|x| !x.is_null())
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| Error::from_reason(format!("Invalid TTS quantization: {e}")))?;
        if let Some(q) = &quant {
            if format != Some(1) {
                return Err(Error::from_reason(
                    "Packed TTS checkpoints require explicit mlx_node_tts_format provenance; convert the official checkpoint with mlx convert",
                ));
            }
            if q.group_size <= 0
                || !matches!(q.bits, 2 | 3 | 4 | 5 | 6 | 8)
                || !matches!(q.mode.as_str(), "affine" | "mxfp4" | "mxfp8")
            {
                return Err(Error::from_reason("Unsupported TTS quantization geometry"));
            }
        }
        let result = Self {
            tensors: load_all_safetensors(path, false)?,
            mlx_layout: format == Some(1),
            quant,
        };
        if result.tensors.is_empty() {
            return Err(Error::from_reason("TTS checkpoint contains no weights"));
        }
        Ok(result)
    }
    pub fn get(&self, name: &str) -> Result<MxArray> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::from_reason(format!("Missing TTS tensor {name}")))
    }
    pub fn optional(&self, name: &str) -> Option<MxArray> {
        self.tensors.get(name).cloned()
    }
    pub fn expect_shape(&self, name: &str, expected: &[i64]) -> Result<()> {
        let actual = self.get(name)?.shape()?;
        if actual.as_ref() != expected {
            return Err(Error::from_reason(format!(
                "TTS tensor {name}: expected {expected:?}, got {:?}",
                actual.as_ref()
            )));
        }
        Ok(())
    }
    pub fn expect_linear(&self, prefix: &str, output: i64, input: i64) -> Result<()> {
        if let Some(scales) = self.optional(&format!("{prefix}.scales")) {
            let q = self
                .quant
                .as_ref()
                .ok_or_else(|| Error::from_reason("Missing packed weight metadata"))?;
            let expected = [output, input / i64::from(q.group_size)];
            if input % i64::from(q.group_size) != 0 || scales.shape()?.as_ref() != expected {
                return Err(Error::from_reason(format!(
                    "TTS linear {prefix}: quantized scale shape does not match configuration"
                )));
            }
            self.expect_shape(
                &format!("{prefix}.weight"),
                &[output, input * i64::from(q.bits) / 32],
            )?;
        } else {
            self.expect_shape(&format!("{prefix}.weight"), &[output, input])?;
        }
        if self.optional(&format!("{prefix}.bias")).is_some() {
            self.expect_shape(&format!("{prefix}.bias"), &[output])?;
        }
        Ok(())
    }
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        let weight = self.get(&format!("{prefix}.weight"))?;
        let bias = self.optional(&format!("{prefix}.bias"));
        let mut layer = Linear::from_weights(&weight, bias.as_ref())?;
        if let Some(scales) = self.optional(&format!("{prefix}.scales")) {
            let quant = self.quant.as_ref().ok_or_else(|| {
                Error::from_reason("Packed TTS weight without quantization metadata")
            })?;
            layer.load_quantized_mode(
                &weight,
                &scales,
                self.optional(&format!("{prefix}.biases")).as_ref(),
                quant.group_size,
                quant.bits,
                &quant.mode,
            )?;
        }
        Ok(layer)
    }
    pub fn rms(&self, prefix: &str, eps: f64) -> Result<RMSNorm> {
        RMSNorm::from_weight(&self.get(&format!("{prefix}.weight"))?, Some(eps))
    }
    pub fn norm(&self, prefix: &str, eps: f64) -> Result<LayerNorm> {
        LayerNorm::from_weights(
            &self.get(&format!("{prefix}.weight"))?,
            self.optional(&format!("{prefix}.bias")).as_ref(),
            Some(eps),
        )
    }
    pub fn conv(&self, prefix: &str, transpose: bool) -> Result<(MxArray, Option<MxArray>)> {
        let weight = self.get(&format!("{prefix}.weight"))?;
        let shape = weight.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "{prefix} must be a 1D convolution"
            )));
        }
        let weight = if self.mlx_layout {
            weight
        } else {
            weight.transpose(Some(if transpose { &[1, 2, 0] } else { &[0, 2, 1] }))?
        };
        Ok((weight, self.optional(&format!("{prefix}.bias"))))
    }
    pub fn materialize(&self) -> Result<()> {
        for chunk in self.tensors.values().collect::<Vec<_>>().chunks(64) {
            MxArray::eval_arrays(chunk)?;
        }
        Ok(())
    }
    pub fn bytes(&self) -> Result<u64> {
        self.tensors.values().try_fold(0u64, |sum, x| {
            Ok(sum + x.size()? * x.dtype()?.byte_size() as u64)
        })
    }
}
