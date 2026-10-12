use super::config::SpeakerConfig;
use super::weights::Weights;
use crate::{
    array::{DType, MxArray},
    audio::dsp::{mel_filter_bank, reflect_index},
    nn::{Activations, Conv1d},
};
use napi::{Error, Result};
use rustfft::{FftPlanner, num_complex::Complex32};

struct Tdnn {
    conv: Conv1d,
    pad: usize,
}
impl Tdnn {
    fn load(w: &Weights, p: &str, dilation: u32) -> Result<Self> {
        let (weight, bias) = w.conv(&format!("{p}.conv"), false)?;
        let pad = (weight.shape()?[1] as usize - 1) * dilation as usize / 2;
        Ok(Self {
            conv: Conv1d::from_weights(
                &weight,
                bias.as_ref(),
                Some(1),
                Some(0),
                Some(dilation),
                Some(1),
            )?,
            pad,
        })
    }
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let n = x.shape()?[1] as usize;
        let x = if self.pad > 0 {
            let indices: Vec<i32> = (0..n + 2 * self.pad)
                .map(|i| reflect_index(i as isize - self.pad as isize, n) as i32)
                .collect();
            x.take(&MxArray::from_int32(&indices, &[indices.len() as i64])?, 1)?
        } else {
            x.clone()
        };
        Activations::relu(&self.conv.forward(&x)?)
    }
}
struct Res2 {
    first: Tdnn,
    second: Tdnn,
    parts: Vec<Tdnn>,
    se1: Conv1d,
    se2: Conv1d,
    scale: usize,
}
impl Res2 {
    fn load(w: &Weights, p: &str, dilation: u32, scale: usize) -> Result<Self> {
        Ok(Self {
            first: Tdnn::load(w, &format!("{p}.tdnn1"), 1)?,
            second: Tdnn::load(w, &format!("{p}.tdnn2"), 1)?,
            parts: (0..scale - 1)
                .map(|i| Tdnn::load(w, &format!("{p}.res2net_block.blocks.{i}"), dilation))
                .collect::<Result<_>>()?,
            se1: super::codec::conv(w, &format!("{p}.se_block.conv1"), 1, 1, 1)?,
            se2: super::codec::conv(w, &format!("{p}.se_block.conv2"), 1, 1, 1)?,
            scale,
        })
    }
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let h = self.first.forward(x)?;
        let width = h.shape()?[2] / self.scale as i64;
        let mut chunks = vec![h.slice_axis(2, 0, width)?];
        for (i, layer) in self.parts.iter().enumerate() {
            let mut part = h.slice_axis(2, (i as i64 + 1) * width, (i as i64 + 2) * width)?;
            if i > 0 {
                part = part.add(&chunks[i])?;
            }
            chunks.push(layer.forward(&part)?);
        }
        let h = self
            .second
            .forward(&super::model::cat(&chunks.iter().collect::<Vec<_>>(), 2)?)?;
        let mean = h.mean(Some(&[1]), Some(true))?;
        let se = Activations::sigmoid(
            &self
                .se2
                .forward(&Activations::relu(&self.se1.forward(&mean)?)?)?,
        )?;
        x.add(&h.mul(&se)?)
    }
}
pub struct SpeakerEncoder {
    pub config: SpeakerConfig,
    first: Tdnn,
    blocks: Vec<Res2>,
    mfa: Tdnn,
    attention: Tdnn,
    attention_out: Conv1d,
    output: Conv1d,
    dtype: DType,
}
impl SpeakerEncoder {
    pub fn load(w: &Weights, value: serde_json::Value) -> Result<Self> {
        let c: SpeakerConfig =
            serde_json::from_value(value).map_err(|e| Error::from_reason(e.to_string()))?;
        if c.enc_channels.len() < 3
            || c.enc_dilations.len() != c.enc_channels.len()
            || c.enc_kernel_sizes.len() != c.enc_channels.len()
            || c.enc_res2net_scale < 2
            || c.mel_dim == 0
            || c.sample_rate == 0
            || c.fft_size == 0
            || c.hop_size == 0
            || c.hop_size > c.fft_size
            || c.enc_channels.contains(&0)
            || c.enc_kernel_sizes.contains(&0)
            || c.enc_dilations.contains(&0)
        {
            return Err(Error::from_reason("Invalid speaker encoder configuration"));
        }
        let (first_weight, _) = w.conv("speaker_encoder.blocks.0.conv", false)?;
        let (output_weight, _) = w.conv("speaker_encoder.fc", false)?;
        if first_weight.shape()?.as_ref()
            != [
                c.enc_channels[0] as i64,
                c.enc_kernel_sizes[0] as i64,
                c.mel_dim as i64,
            ]
            || output_weight.shape()?.as_ref()
                != [
                    c.enc_dim as i64,
                    1,
                    (2 * c.enc_channels[c.enc_channels.len() - 1]) as i64,
                ]
        {
            return Err(Error::from_reason(
                "Speaker encoder weights differ from configuration",
            ));
        }
        let first = Tdnn::load(w, "speaker_encoder.blocks.0", c.enc_dilations[0])?;
        let blocks = (1..c.enc_channels.len() - 1)
            .map(|i| {
                Res2::load(
                    w,
                    &format!("speaker_encoder.blocks.{i}"),
                    c.enc_dilations[i],
                    c.enc_res2net_scale,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            config: c,
            first,
            blocks,
            mfa: Tdnn::load(w, "speaker_encoder.mfa", 1)?,
            attention: Tdnn::load(w, "speaker_encoder.asp.tdnn", 1)?,
            attention_out: super::codec::conv(w, "speaker_encoder.asp.conv", 1, 1, 1)?,
            output: super::codec::conv(w, "speaker_encoder.fc", 1, 1, 1)?,
            dtype: w.get("speaker_encoder.blocks.0.conv.weight")?.dtype()?,
        })
    }
    pub fn encode(&self, audio: &[f32]) -> Result<MxArray> {
        let mel = speaker_mel(audio, &self.config)?;
        let mut x = self.first.forward(&mel.astype(self.dtype)?)?;
        let mut stages = Vec::new();
        for block in &self.blocks {
            x = block.forward(&x)?;
            stages.push(x.clone());
        }
        x = self
            .mfa
            .forward(&super::model::cat(&stages.iter().collect::<Vec<_>>(), 2)?)?;
        let shape = x.shape()?;
        let mean = x.mean(Some(&[1]), Some(true))?;
        let std = x
            .sub(&mean)?
            .square()?
            .mean(Some(&[1]), Some(true))?
            .add_scalar(1e-12)?
            .sqrt()?;
        let context = super::model::cat(
            &[&x, &mean.broadcast_to(&shape)?, &std.broadcast_to(&shape)?],
            2,
        )?;
        let attn = Activations::softmax(
            &self
                .attention_out
                .forward(&self.attention.forward(&context)?.tanh()?)?,
            Some(1),
        )?;
        let mean = attn.mul(&x)?.sum(Some(&[1]), Some(true))?;
        let std = attn
            .mul(&x.sub(&mean)?.square()?)?
            .sum(Some(&[1]), Some(true))?
            .clip(Some(1e-12), None)?
            .sqrt()?;
        self.output.forward(&super::model::cat(&[&mean, &std], 2)?)
    }
}

/// Speaker encoder training frontend: periodic Hann, manual reflect padding,
/// magnitude spectrum, Slaney filters, natural log. ASR uses a different policy.
pub(super) fn speaker_mel(audio: &[f32], config: &SpeakerConfig) -> Result<MxArray> {
    let (fft_size, hop_size, sample_rate, mels) = (
        config.fft_size,
        config.hop_size,
        config.sample_rate,
        config.mel_dim,
    );
    let (padding, frames) = speaker_geometry(audio.len(), config)?;
    if audio.iter().any(|x| !x.is_finite()) {
        return Err(Error::from_reason(
            "Reference audio is too short or contains non-finite samples",
        ));
    }
    let filters = mel_filter_bank(
        fft_size / 2 + 1,
        mels,
        0.,
        sample_rate as f32 / 2.,
        sample_rate as f32,
    );
    let fft = FftPlanner::<f32>::new().plan_fft_forward(fft_size);
    let mut buf = vec![Complex32::default(); fft_size];
    let window: Vec<f32> = (0..fft_size)
        .map(|i| {
            (0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / fft_size as f64).cos()) as f32
        })
        .collect();
    let mut magnitude = vec![0f32; fft_size / 2 + 1];
    let mut features = vec![0f32; frames * mels];
    for frame in 0..frames {
        for (i, x) in buf.iter_mut().enumerate() {
            let idx = reflect_index(
                (frame * hop_size + i) as isize - padding as isize,
                audio.len(),
            );
            *x = Complex32::new(audio[idx] * window[i], 0.);
        }
        fft.process(&mut buf);
        for (value, bin) in magnitude.iter_mut().zip(&buf) {
            *value = (bin.norm_sqr() + 1e-9).sqrt();
        }
        for m in 0..mels {
            let value: f32 = (0..fft_size / 2 + 1)
                .map(|f| filters[f * mels + m] * magnitude[f])
                .sum();
            features[frame * mels + m] = value.max(1e-5).ln();
        }
    }
    MxArray::from_float32(&features, &[1, frames as i64, mels as i64])
}

fn speaker_geometry(length: usize, c: &SpeakerConfig) -> Result<(usize, usize)> {
    if c.fft_size < 2
        || !c.fft_size.is_multiple_of(2)
        || c.hop_size == 0
        || c.hop_size > c.fft_size
        || c.mel_dim == 0
        || c.sample_rate == 0
    {
        return Err(Error::from_reason("Invalid speaker frontend geometry"));
    }
    let padding = (c.fft_size - c.hop_size) / 2;
    let available = length
        .checked_add(2 * padding)
        .and_then(|n| n.checked_sub(c.fft_size));
    let Some(available) = available.filter(|_| length > padding) else {
        return Err(Error::from_reason(
            "Reference audio is too short for the speaker frontend",
        ));
    };
    Ok((padding, available / c.hop_size + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen3_tts_speaker_frontend_rejects_short_and_invalid_geometry() {
        let mut c = SpeakerConfig::default();
        assert!(speaker_geometry(100, &c).is_err());
        c.hop_size = c.fft_size;
        assert!(speaker_geometry(100, &c).is_err());
        assert_eq!(speaker_geometry(c.fft_size, &c).unwrap(), (0, 1));
        c.fft_size = 1;
        c.hop_size = 1;
        assert!(speaker_geometry(100, &c).is_err());
        c.fft_size = 1023;
        assert!(speaker_geometry(10000, &c).is_err());
    }
}
