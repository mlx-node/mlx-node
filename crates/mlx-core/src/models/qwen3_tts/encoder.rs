//! Reference-audio codec encoder. Original checkpoint names are preserved.
use super::{
    config::{CodecConfig, TransformerConfig},
    model::cat,
    transformer::Decoder,
    weights::Weights,
};
use crate::{
    array::{DType, MxArray},
    nn::Conv1d,
};
use napi::{Error, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct EncoderConfig {
    #[serde(flatten)]
    transformer: TransformerConfig,
    num_residual_layers: usize,
    dilation_growth_rate: u32,
    upsampling_ratios: Vec<u32>,
    use_causal_conv: bool,
    use_conv_shortcut: bool,
    num_semantic_quantizers: usize,
}
struct Causal {
    conv: Conv1d,
    padding: i32,
    stride: i32,
    edge: bool,
}
impl Causal {
    fn load(w: &Weights, p: &str, stride: u32, dilation: u32, edge: bool) -> Result<Self> {
        let (weight, bias) = w.conv(p, false)?;
        let effective = (weight.shape()?[1] - 1) * i64::from(dilation) + 1;
        Ok(Self {
            conv: Conv1d::from_weights(
                &weight,
                bias.as_ref(),
                Some(stride),
                Some(0),
                Some(dilation),
                Some(1),
            )?,
            padding: effective as i32 - stride as i32,
            stride: stride as i32,
            edge,
        })
    }
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        let extra = (self.stride as i64 - shape[1] % self.stride as i64) % self.stride as i64;
        let x = if self.edge {
            let left =
                x.slice_axis(1, 0, 1)?
                    .broadcast_to(&[shape[0], self.padding as i64, shape[2]])?;
            let right = x
                .slice_axis(1, shape[1] - 1, shape[1])?
                .broadcast_to(&[shape[0], extra, shape[2]])?;
            cat(&[&left, x, &right], 1)?
        } else {
            x.pad(&[0, 0, self.padding, extra as i32, 0, 0], 0.)?
        };
        self.conv.forward(&x)
    }
}
fn elu(x: &MxArray) -> Result<MxArray> {
    x.less(&MxArray::scalar_float(0.)?)?
        .where_(&x.exp()?.sub_scalar(1.)?, x)
}
struct Residual {
    first: Causal,
    second: Causal,
    shortcut: Option<Causal>,
}
impl Residual {
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let h = self.first.forward(&elu(x)?)?;
        let h = self.second.forward(&elu(&h)?)?;
        h.add(&match &self.shortcut {
            Some(layer) => layer.forward(x)?,
            None => x.clone(),
        })
    }
}
struct Stage {
    residuals: Vec<Residual>,
    down: Causal,
}
struct Rvq {
    input: Conv1d,
    books: Vec<MxArray>,
}
impl Rvq {
    fn load(w: &Weights, p: &str, count: usize) -> Result<Self> {
        let books = (0..count)
            .map(|i| {
                let p = format!("{p}.layers.{i}.codebook");
                let sum = w.get(&format!("{p}.embed_sum"))?;
                let usage = w.get(&format!("{p}.cluster_usage"))?;
                sum.div(&usage.clip(Some(1e-5), None)?.reshape(&[-1, 1])?)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            input: super::codec::conv(w, &format!("{p}.input_proj"), 1, 1, 1)?,
            books,
        })
    }
    fn encode(&self, x: &MxArray) -> Result<Vec<MxArray>> {
        let mut residual = self.input.forward(x)?.astype(DType::Float32)?;
        let mut codes = Vec::new();
        for book in &self.books {
            let dot = residual.matmul(&book.transpose(Some(&[1, 0]))?)?;
            let norm = book
                .square()?
                .sum(Some(&[-1]), Some(false))?
                .mul_scalar(0.5)?;
            let ids = norm.sub(&dot)?.argmin(-1, Some(false))?;
            residual = residual.sub(&book.take(&ids, 0)?)?;
            codes.push(ids);
        }
        Ok(codes)
    }
}
pub struct CodecEncoder {
    first: Causal,
    stages: Vec<Stage>,
    last: Causal,
    transformer: Decoder,
    down: Causal,
    semantic: Rvq,
    acoustic: Rvq,
    dtype: DType,
    transformer_stride: usize,
    max_position_embeddings: usize,
}

fn validate_audio_length(
    samples: usize,
    source_rate: u32,
    target_rate: u32,
    transformer_stride: usize,
    max_position_embeddings: usize,
) -> Result<()> {
    if source_rate == 0 || target_rate == 0 || transformer_stride == 0 {
        return Err(Error::from_reason("Invalid reference audio geometry"));
    }
    // Match resample_mono's floor, followed by the encoder's causal ceil-strides.
    // Calculate before resampling so oversized references allocate no model input.
    let resampled = samples as u128 * u128::from(target_rate) / u128::from(source_rate);
    validate_encoder_frames(resampled, transformer_stride, max_position_embeddings)
}

fn validate_encoder_frames(
    resampled_samples: u128,
    transformer_stride: usize,
    max_position_embeddings: usize,
) -> Result<()> {
    let frames = resampled_samples.div_ceil(transformer_stride as u128);
    if frames > max_position_embeddings as u128 {
        return Err(Error::from_reason(
            "Reference audio exceeds speech encoder context capacity",
        ));
    }
    Ok(())
}
impl CodecEncoder {
    pub fn load(w: &Weights, config: &CodecConfig) -> Result<Self> {
        let c: EncoderConfig = serde_json::from_value(
            config
                .encoder_config
                .clone()
                .ok_or_else(|| Error::from_reason("Codec has no reference encoder"))?,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        if !c.use_causal_conv
            || c.num_residual_layers == 0
            || c.upsampling_ratios.is_empty()
            || c.upsampling_ratios.contains(&0)
            || c.num_semantic_quantizers == 0
            || c.num_semantic_quantizers >= config.encoder_valid_num_quantizers
        {
            return Err(Error::from_reason("Unsupported reference codec geometry"));
        }
        let mut layer_index = 1;
        let mut stages = Vec::new();
        for &ratio in c.upsampling_ratios.iter().rev() {
            let mut residuals = Vec::new();
            for r in 0..c.num_residual_layers {
                let p = format!("encoder.encoder.layers.{layer_index}");
                residuals.push(Residual {
                    first: Causal::load(
                        w,
                        &format!("{p}.block.1.conv"),
                        1,
                        c.dilation_growth_rate.pow(r as u32),
                        false,
                    )?,
                    second: Causal::load(w, &format!("{p}.block.3.conv"), 1, 1, false)?,
                    shortcut: if c.use_conv_shortcut {
                        Some(Causal::load(w, &format!("{p}.shortcut.conv"), 1, 1, false)?)
                    } else {
                        None
                    },
                });
                layer_index += 1;
            }
            layer_index += 1; // ELU before the strided convolution.
            let down = Causal::load(
                w,
                &format!("encoder.encoder.layers.{layer_index}.conv"),
                ratio,
                1,
                false,
            )?;
            layer_index += 1;
            stages.push(Stage { residuals, down });
        }
        let total = c
            .upsampling_ratios
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
            .ok_or_else(|| Error::from_reason("Encoder stride overflow"))?;
        if !config.encode_downsample_rate.is_multiple_of(total) {
            return Err(Error::from_reason("Encoder frame ratio must be integral"));
        }
        Ok(Self {
            first: Causal::load(w, "encoder.encoder.layers.0.conv", 1, 1, false)?,
            stages,
            last: Causal::load(
                w,
                &format!("encoder.encoder.layers.{}.conv", layer_index + 1),
                1,
                1,
                false,
            )?,
            transformer: Decoder::load_encoder(w, "encoder.encoder_transformer", &c.transformer)?,
            down: Causal::load(
                w,
                "encoder.downsample.conv",
                (config.encode_downsample_rate / total) as u32,
                1,
                true,
            )?,
            semantic: Rvq::load(
                w,
                "encoder.quantizer.semantic_residual_vector_quantizer",
                c.num_semantic_quantizers,
            )?,
            acoustic: Rvq::load(
                w,
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                config.encoder_valid_num_quantizers - c.num_semantic_quantizers,
            )?,
            dtype: w.get("encoder.encoder.layers.0.conv.weight")?.dtype()?,
            transformer_stride: total,
            max_position_embeddings: c.transformer.max_position_embeddings,
        })
    }
    pub fn validate_audio_length(
        &self,
        samples: usize,
        source_rate: u32,
        target_rate: u32,
    ) -> Result<()> {
        validate_audio_length(
            samples,
            source_rate,
            target_rate,
            self.transformer_stride,
            self.max_position_embeddings,
        )
    }
    pub fn encode(&self, audio: &[f32]) -> Result<MxArray> {
        validate_encoder_frames(
            audio.len() as u128,
            self.transformer_stride,
            self.max_position_embeddings,
        )?;
        let mut x = self.first.forward(
            &MxArray::from_float32(audio, &[1, audio.len() as i64, 1])?.astype(self.dtype)?,
        )?;
        for stage in &self.stages {
            for residual in &stage.residuals {
                x = residual.forward(&x)?;
            }
            x = stage.down.forward(&elu(&x)?)?;
        }
        x = self.last.forward(&elu(&x)?)?;
        x = self
            .transformer
            .forward(&x, &mut self.transformer.state())?;
        x = self.down.forward(&x)?;
        let mut codes = self.semantic.encode(&x)?;
        codes.extend(self.acoustic.encode(&x)?);
        MxArray::stack(codes.iter().collect(), Some(2))
    }
}

#[cfg(test)]
mod tests {
    use super::validate_audio_length;

    #[test]
    fn qwen3_tts_reference_context_uses_resampled_encoder_rows() {
        let limit = 960 * 8000;
        assert!(validate_audio_length(limit, 24000, 24000, 960, 8000).is_ok());
        assert!(validate_audio_length(limit + 1, 24000, 24000, 960, 8000).is_err());
        assert!(validate_audio_length(limit * 2 + 1, 48000, 24000, 960, 8000).is_ok());
        assert!(validate_audio_length(limit * 2 + 2, 48000, 24000, 960, 8000).is_err());
        assert!(validate_audio_length(usize::MAX, 1, u32::MAX, 960, 8000).is_err());
        assert!(validate_audio_length(1, 0, 24000, 960, 8000).is_err());
    }
}
