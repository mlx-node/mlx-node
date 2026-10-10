//! Qwen speech tokenizer decoder; explicit per-stream state throughout.
use super::{
    config::CodecConfig,
    transformer::{AttentionState, Decoder},
    weights::Weights,
};
use crate::{
    array::{DType, MxArray},
    nn::{
        Activations, Conv1d, LayerNorm, Linear,
        causal_conv::{CausalConv1d, CausalTransposeConv1d, ConvState},
    },
};
use napi::{Error, Result};
use std::{collections::HashMap, sync::atomic::AtomicBool};

/// The decoder is an F32 architecture: half-precision weights run slower
/// (bf16 conv/SDPA kernels) and measurably degrade output (~34 dB SNR loss,
/// PR-204 audit #5). Fail loudly instead of silently degrading.
fn check_decoder_dtype(tensors: &HashMap<String, MxArray>) -> Result<()> {
    for (name, tensor) in tensors {
        if name.starts_with("decoder.") && tensor.dtype()? != DType::Float32 {
            return Err(Error::from_reason(format!(
                "Codec decoder tensor {name} must be float32 (got {:?})",
                tensor.dtype()?
            )));
        }
    }
    Ok(())
}

pub fn conv(w: &Weights, prefix: &str, stride: u32, dilation: u32, groups: u32) -> Result<Conv1d> {
    let (weight, bias) = w.conv(prefix, false)?;
    Conv1d::from_weights(
        &weight,
        bias.as_ref(),
        Some(stride),
        Some(0),
        Some(dilation),
        Some(groups),
    )
}
fn causal(w: &Weights, prefix: &str, dilation: u32, groups: u32) -> Result<CausalConv1d> {
    let (weight, bias) = w.conv(prefix, false)?;
    CausalConv1d::new(&weight, bias.as_ref(), dilation, groups)
}
fn transpose(w: &Weights, prefix: &str, stride: i32) -> Result<CausalTransposeConv1d> {
    let (weight, bias) = w.conv(prefix, true)?;
    CausalTransposeConv1d::new(weight, bias, stride)
}

pub fn codebook(w: &Weights, prefix: &str) -> Result<MxArray> {
    let sum = w.get(&format!("{prefix}.embedding_sum"))?;
    let usage = w.get(&format!("{prefix}.cluster_usage"))?;
    let size = usage.shape()?[0];
    sum.div(&usage.clip(Some(1e-5), None)?.reshape(&[size, 1])?)
}

struct Rvq {
    books: Vec<MxArray>,
    output: Conv1d,
}
impl Rvq {
    fn load(w: &Weights, prefix: &str, count: usize) -> Result<Self> {
        let books = (0..count)
            .map(|i| codebook(w, &format!("{prefix}.vq.layers.{i}._codebook")))
            .collect::<Result<_>>()?;
        Ok(Self {
            books,
            output: conv(w, &format!("{prefix}.output_proj"), 1, 1, 1)?,
        })
    }
    fn decode(&self, codes: &MxArray, start: usize) -> Result<MxArray> {
        let shape = codes.shape()?;
        let mut sum = None;
        for (index, book) in self.books.iter().enumerate() {
            let ids = codes
                .slice_axis(2, (start + index) as i64, (start + index + 1) as i64)?
                .reshape(&[shape[0], shape[1]])?;
            let x = book.take(&ids, 0)?;
            sum = Some(match sum {
                Some(prev) => MxArray::add(&prev, &x)?,
                None => x,
            });
        }
        self.output
            .forward(&sum.ok_or_else(|| Error::from_reason("Empty RVQ"))?)
    }
}

struct Snake {
    alpha: MxArray,
    beta: MxArray,
}
impl Snake {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            alpha: w.get(&format!("{prefix}.alpha"))?.reshape(&[-1])?.exp()?,
            beta: w
                .get(&format!("{prefix}.beta"))?
                .reshape(&[-1])?
                .exp()?
                .add_scalar(1e-9)?,
        })
    }
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        x.add(&x.mul(&self.alpha)?.sin()?.square()?.div(&self.beta)?)
    }
}
struct Residual {
    first: Snake,
    conv1: CausalConv1d,
    second: Snake,
    conv2: CausalConv1d,
}
#[derive(Default, Clone)]
struct ResidualState {
    first: ConvState,
    second: ConvState,
}
impl Residual {
    fn load(w: &Weights, prefix: &str, dilation: u32) -> Result<Self> {
        Ok(Self {
            first: Snake::load(w, &format!("{prefix}.act1"))?,
            conv1: causal(w, &format!("{prefix}.conv1.conv"), dilation, 1)?,
            second: Snake::load(w, &format!("{prefix}.act2"))?,
            conv2: causal(w, &format!("{prefix}.conv2.conv"), 1, 1)?,
        })
    }
    fn step(&self, x: &MxArray, state: &mut ResidualState) -> Result<MxArray> {
        let h = self.conv1.step(&self.first.forward(x)?, &mut state.first)?;
        x.add(
            &self
                .conv2
                .step(&self.second.forward(&h)?, &mut state.second)?,
        )
    }
}
struct DecoderBlock {
    act: Snake,
    up: CausalTransposeConv1d,
    residuals: Vec<Residual>,
}
#[derive(Clone)]
struct BlockState {
    up: ConvState,
    residuals: Vec<ResidualState>,
}
impl DecoderBlock {
    fn step(&self, x: &MxArray, state: &mut BlockState) -> Result<MxArray> {
        let mut x = self.up.step(&self.act.forward(x)?, &mut state.up)?;
        for (layer, state) in self.residuals.iter().zip(&mut state.residuals) {
            x = layer.step(&x, state)?;
        }
        Ok(x)
    }
}
struct ConvNext {
    depthwise: CausalConv1d,
    norm: LayerNorm,
    first: Linear,
    second: Linear,
    gamma: MxArray,
}
impl ConvNext {
    fn load(w: &Weights, prefix: &str, dim: i64, eps: f64) -> Result<Self> {
        Ok(Self {
            depthwise: causal(w, &format!("{prefix}.dwconv.conv"), 1, dim as u32)?,
            norm: w.norm(&format!("{prefix}.norm"), eps)?,
            first: w.linear(&format!("{prefix}.pwconv1"))?,
            second: w.linear(&format!("{prefix}.pwconv2"))?,
            gamma: w.get(&format!("{prefix}.gamma"))?,
        })
    }
    fn step(&self, x: &MxArray, state: &mut ConvState) -> Result<MxArray> {
        let h = self.norm.forward(&self.depthwise.step(x, state)?)?;
        x.add(
            &self
                .second
                .forward(&Activations::gelu_exact(&self.first.forward(&h)?)?)?
                .mul(&self.gamma)?,
        )
    }
}

pub struct CodecDecoder {
    pub config: CodecConfig,
    semantic: Rvq,
    acoustic: Rvq,
    pre: CausalConv1d,
    transformer: Decoder,
    input: Linear,
    output: Linear,
    upsample: Vec<(CausalTransposeConv1d, ConvNext)>,
    initial: CausalConv1d,
    blocks: Vec<DecoderBlock>,
    final_act: Snake,
    final_conv: CausalConv1d,
}
pub struct CodecState {
    pre: ConvState,
    transformer: Vec<AttentionState>,
    upsample: Vec<(ConvState, ConvState)>,
    initial: ConvState,
    blocks: Vec<BlockState>,
    final_conv: ConvState,
}
impl CodecState {
    /// Reuse immutable convolution history while isolating mutable KV buffers.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            pre: self.pre.clone(),
            transformer: self
                .transformer
                .iter()
                .map(|state| {
                    Ok(AttentionState {
                        kv: state.kv.fork()?,
                        position: state.position,
                    })
                })
                .collect::<Result<_>>()?,
            upsample: self.upsample.clone(),
            initial: self.initial.clone(),
            blocks: self.blocks.clone(),
            final_conv: self.final_conv.clone(),
        })
    }

    fn retained(&self) -> Vec<&MxArray> {
        let mut arrays = Vec::new();
        arrays.extend(self.pre.history.as_ref());
        for state in &self.transformer {
            arrays.extend(state.kv.keys_ref());
            arrays.extend(state.kv.values_ref());
        }
        for (up, next) in &self.upsample {
            arrays.extend(up.history.as_ref());
            arrays.extend(next.history.as_ref());
        }
        arrays.extend(self.initial.history.as_ref());
        for block in &self.blocks {
            arrays.extend(block.up.history.as_ref());
            for residual in &block.residuals {
                arrays.extend(residual.first.history.as_ref());
                arrays.extend(residual.second.history.as_ref());
            }
        }
        arrays.extend(self.final_conv.history.as_ref());
        arrays
    }

    /// Materialize every retained tail, including histories not needed by the
    /// most recent PCM output, so a prepared prefix retains no growing graph.
    fn eval(&self) -> Result<()> {
        MxArray::eval_arrays(&self.retained())
    }

    /// Submit the same materialization without waiting, so the next chunk's
    /// graph build overlaps the GPU eval. The final `eval` still performs the
    /// one synchronous wait.
    fn eval_async(&self) {
        MxArray::async_eval_arrays(&self.retained());
    }
}
impl CodecDecoder {
    pub fn load(w: &Weights, config: CodecConfig) -> Result<Self> {
        let c = &config.decoder_config;
        for rvq in ["rvq_first", "rvq_rest"] {
            let (weight, _) = w.conv(&format!("decoder.quantizer.{rvq}.output_proj"), false)?;
            if weight.shape()?.as_ref() != [c.codebook_dim, 1, c.codebook_dim / 2] {
                return Err(Error::from_reason(
                    "Codec projection shape differs from configuration",
                ));
            }
        }
        check_decoder_dtype(&w.tensors)?;
        let (initial, _) = w.conv("decoder.decoder.0.conv", false)?;
        if initial.shape()?[0] != c.decoder_dim || initial.shape()?[2] != c.latent_dim {
            return Err(Error::from_reason(
                "Codec decoder width differs from configuration",
            ));
        }
        let transformer = Decoder::load(w, "decoder.pre_transformer", &c.transformer, false, true)?;
        let upsample = c
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                Ok((
                    transpose(w, &format!("decoder.upsample.{i}.0.conv"), rate)?,
                    ConvNext::load(
                        w,
                        &format!("decoder.upsample.{i}.1"),
                        c.latent_dim,
                        c.convnext_norm_eps,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        let blocks = c
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                let p = format!("decoder.decoder.{}.block", i + 1);
                // DAC residual dilation schedule is part of this tokenizer architecture.
                let residuals = c
                    .residual_dilations
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(j, d)| Residual::load(w, &format!("{p}.{}", j + 2), d))
                    .collect::<Result<_>>()?;
                Ok(DecoderBlock {
                    act: Snake::load(w, &format!("{p}.0"))?,
                    up: transpose(w, &format!("{p}.1.conv"), rate)?,
                    residuals,
                })
            })
            .collect::<Result<_>>()?;
        let last = c.upsample_rates.len() + 1;
        Ok(Self {
            semantic: Rvq::load(w, "decoder.quantizer.rvq_first", c.num_semantic_quantizers)?,
            acoustic: Rvq::load(
                w,
                "decoder.quantizer.rvq_rest",
                c.num_quantizers - c.num_semantic_quantizers,
            )?,
            pre: causal(w, "decoder.pre_conv.conv", 1, 1)?,
            transformer,
            input: w.linear("decoder.pre_transformer.input_proj")?,
            output: w.linear("decoder.pre_transformer.output_proj")?,
            upsample,
            initial: causal(w, "decoder.decoder.0.conv", 1, 1)?,
            blocks,
            final_act: Snake::load(w, &format!("decoder.decoder.{last}"))?,
            final_conv: causal(w, &format!("decoder.decoder.{}.conv", last + 1), 1, 1)?,
            config,
        })
    }
    pub fn state(&self) -> CodecState {
        CodecState {
            pre: ConvState::default(),
            transformer: self.transformer.state(),
            upsample: self
                .upsample
                .iter()
                .map(|_| (ConvState::default(), ConvState::default()))
                .collect(),
            initial: ConvState::default(),
            blocks: self
                .blocks
                .iter()
                .map(|b| BlockState {
                    up: ConvState::default(),
                    residuals: b
                        .residuals
                        .iter()
                        .map(|_| ResidualState::default())
                        .collect(),
                })
                .collect(),
            final_conv: ConvState::default(),
        }
    }
    /// Codes use [batch, frames, codebooks]; the count is independent of code values.
    pub fn step(&self, codes: &MxArray, state: &mut CodecState) -> Result<MxArray> {
        let shape = codes.shape()?;
        if shape.len() != 3
            || shape[2] != self.config.decoder_config.num_quantizers as i64
            || shape[1] == 0
        {
            return Err(Error::from_reason("Invalid speech codec token shape"));
        }
        let x = self.semantic.decode(codes, 0)?.add(
            &self
                .acoustic
                .decode(codes, self.config.decoder_config.num_semantic_quantizers)?,
        )?;
        let x = self.pre.step(&x, &mut state.pre)?;
        let x = self
            .transformer
            .forward(&self.input.forward(&x)?, &mut state.transformer)?;
        let mut x = self.output.forward(&x)?;
        for ((up, next), (up_state, next_state)) in self.upsample.iter().zip(&mut state.upsample) {
            x = next.step(&up.step(&x, up_state)?, next_state)?;
        }
        x = self.initial.step(&x, &mut state.initial)?;
        for (block, state) in self.blocks.iter().zip(&mut state.blocks) {
            x = block.step(&x, state)?;
        }
        let x = self
            .final_conv
            .step(&self.final_act.forward(&x)?, &mut state.final_conv)?
            .clip(Some(-1.), Some(1.))?;
        if x.shape()?[1] != shape[1] * self.config.decode_upsample_rate as i64 {
            return Err(Error::from_reason(
                "Codec output length differs from configured upsample rate",
            ));
        }
        x.astype(DType::Float32)?.reshape(&[-1])
    }
    #[cfg(test)]
    pub fn decode(&self, codes: &MxArray) -> Result<MxArray> {
        self.step(codes, &mut self.state())
    }

    /// Prepare a reusable reference prefix without retaining its PCM or a
    /// reference-length decoder graph. One codec frame bounds temporary memory.
    /// `step` must stay per-frame: multi-frame shapes change conv/GEMM tiling
    /// and are not bit-identical to the 1-frame loop (chunked runs verified
    /// against this loop diverge in sample 0). The history materialization is
    /// submitted asynchronously so each frame's graph build overlaps the GPU
    /// eval of the previous one, then the final `eval` performs the one
    /// synchronous wait that bounds retained state and surfaces eval errors.
    pub fn prepare_prefix(&self, codes: &MxArray, cancelled: &AtomicBool) -> Result<CodecState> {
        let mut state = self.state();
        for frame in 0..codes.shape_at(1)? {
            super::check_cancelled(cancelled)?;
            let _ = self.step(&codes.slice_axis(1, frame, frame + 1)?, &mut state)?;
            state.eval_async();
        }
        super::check_cancelled(cancelled)?;
        state.eval()?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_dtype_guard_rejects_half_weights() {
        let tensors = |decoder_dtype: DType| {
            HashMap::from([
                (
                    "decoder.pre_transformer.input_proj.weight".to_string(),
                    MxArray::from_float32(&[0.; 4], &[2, 2])
                        .unwrap()
                        .astype(decoder_dtype)
                        .unwrap(),
                ),
                (
                    "decoder.decoder.0.conv.weight".to_string(),
                    MxArray::from_float32(&[0.; 8], &[2, 2, 2])
                        .unwrap()
                        .astype(decoder_dtype)
                        .unwrap(),
                ),
                // The encoder is a separate family with its own checks.
                (
                    "encoder.encoder.0.conv.weight".to_string(),
                    MxArray::from_float32(&[0.; 4], &[2, 2]).unwrap(),
                ),
            ])
        };
        check_decoder_dtype(&tensors(DType::Float32)).unwrap();
        let err = check_decoder_dtype(&tensors(DType::BFloat16)).unwrap_err();
        assert!(err.reason.contains("must be float32"), "{err:?}");
    }

    /// `prepare_prefix` must retain bit-identical state: async-evaluating the
    /// histories between frames cannot change the continuation's PCM bits.
    #[test]
    #[ignore = "Requires a Qwen3-TTS checkpoint in TTS_TEST_MODEL"]
    fn prepare_prefix_matches_per_frame() {
        use super::super::config::{CodecConfig, read_json};
        use super::super::weights::Weights;
        use std::path::Path;

        let root = Path::new(&std::env::var("TTS_TEST_MODEL").unwrap()).join("speech_tokenizer");
        let weights = Weights::load(&root).unwrap();
        let config: CodecConfig = read_json(&root.join("config.json")).unwrap();
        let codec = CodecDecoder::load(&weights, config).unwrap();
        let groups = codec.config.decoder_config.num_quantizers as i64;
        // 19 prefix frames fill the state; 4 continuation frames decode
        // through it so history differences surface in the PCM bits.
        let values: Vec<i32> = (0..23 * groups)
            .map(|i| ((i * 37 + 11) % 1024) as i32)
            .collect();
        let codes = MxArray::from_int32(&values, &[1, 23, groups]).unwrap();
        let prefix = codes.slice_axis(1, 0, 19).unwrap();
        let continuation = codes.slice_axis(1, 19, 23).unwrap();
        let prepared = codec
            .prepare_prefix(&prefix, &AtomicBool::new(false))
            .unwrap();
        let mut legacy = codec.state();
        for frame in 0..19i64 {
            let _ = codec
                .step(
                    &prefix.slice_axis(1, frame, frame + 1).unwrap(),
                    &mut legacy,
                )
                .unwrap();
            legacy.eval().unwrap();
        }
        let decode = |mut state: CodecState| {
            codec
                .step(&continuation, &mut state)
                .unwrap()
                .to_float32()
                .unwrap()
        };
        let a = decode(prepared.fork().unwrap());
        let b = decode(legacy);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "sample {i}: prepared prefix changed PCM bits"
            );
        }
    }
}
