use super::{config::TransformerConfig, weights::Weights};
use crate::{
    array::{DType, MxArray, scaled_dot_product_attention, scaled_dot_product_attention_causal},
    nn::{Activations, LayerNorm, Linear, RMSNorm, RoPE},
    transformer::{KVCache, kv_cache::KvBlock},
};
use napi::{Error, Result};
use std::sync::OnceLock;

#[derive(Default)]
pub struct AttentionState {
    pub kv: KVCache,
    pub position: usize,
}

enum Norm {
    Rms(RMSNorm),
    Layer(LayerNorm),
}
impl Norm {
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
        }
    }
}

pub struct DecoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    norm1: Norm,
    norm2: Norm,
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    attn_scale: Option<MxArray>,
    mlp_scale: Option<MxArray>,
    config: TransformerConfig,
    /// Effective attention policy can differ from checkpoint metadata: the
    /// offline Mimi encoder supplies a full causal mask explicitly.
    attention_window: Option<usize>,
    rope: RoPE,
}
impl DecoderLayer {
    pub fn load(
        w: &Weights,
        prefix: &str,
        c: &TransformerConfig,
        qk_norm: bool,
        layer_scale: bool,
    ) -> Result<Self> {
        c.validate()?;
        if c.hidden_act != "silu" {
            return Err(Error::from_reason("Unsupported TTS decoder activation"));
        }
        let attn = format!("{prefix}.self_attn");
        for (name, out, input) in [
            ("q_proj", c.num_attention_heads * c.head_dim, c.hidden_size),
            ("k_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
            ("v_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
            ("o_proj", c.hidden_size, c.num_attention_heads * c.head_dim),
        ] {
            w.expect_linear(&format!("{attn}.{name}"), out, input)?;
            if c.attention_bias {
                w.expect_shape(&format!("{attn}.{name}.bias"), &[out])?;
            }
        }
        for name in ["input_layernorm", "post_attention_layernorm"] {
            w.expect_shape(&format!("{prefix}.{name}.weight"), &[c.hidden_size])?;
        }
        for name in ["gate_proj", "up_proj"] {
            w.expect_linear(
                &format!("{prefix}.mlp.{name}"),
                c.intermediate_size,
                c.hidden_size,
            )?;
        }
        w.expect_linear(
            &format!("{prefix}.mlp.down_proj"),
            c.hidden_size,
            c.intermediate_size,
        )?;
        Ok(Self {
            q: w.linear(&format!("{attn}.q_proj"))?,
            k: w.linear(&format!("{attn}.k_proj"))?,
            v: w.linear(&format!("{attn}.v_proj"))?,
            o: w.linear(&format!("{attn}.o_proj"))?,
            q_norm: if qk_norm {
                Some(w.rms(&format!("{attn}.q_norm"), c.rms_norm_eps)?)
            } else {
                None
            },
            k_norm: if qk_norm {
                Some(w.rms(&format!("{attn}.k_norm"), c.rms_norm_eps)?)
            } else {
                None
            },
            norm1: Norm::Rms(w.rms(&format!("{prefix}.input_layernorm"), c.rms_norm_eps)?),
            norm2: Norm::Rms(w.rms(
                &format!("{prefix}.post_attention_layernorm"),
                c.rms_norm_eps,
            )?),
            gate: Some(w.linear(&format!("{prefix}.mlp.gate_proj"))?),
            up: w.linear(&format!("{prefix}.mlp.up_proj"))?,
            down: w.linear(&format!("{prefix}.mlp.down_proj"))?,
            attn_scale: if layer_scale {
                Some(w.get(&format!("{prefix}.self_attn_layer_scale.scale"))?)
            } else {
                None
            },
            mlp_scale: if layer_scale {
                Some(w.get(&format!("{prefix}.mlp_layer_scale.scale"))?)
            } else {
                None
            },
            config: c.clone(),
            attention_window: c.sliding_window,
            rope: RoPE::new(c.head_dim as i32, Some(false), Some(c.rope_theta), None),
        })
    }

    /// `rope(q_norm(q))`, `rope(k_norm(k))` in one Metal dispatch
    /// (`mlx_qk_norm_rope`, bit-identical to the `rms_norm` -> transpose ->
    /// `rope` chain) from `[B, T, H, D]` inputs (any strides) to `[B, H, T, D]`
    /// outputs. `None` on a contract miss (non-Metal, traditional rope, no
    /// q/k norms, differing eps); callers keep the four-op chain.
    fn fused_qk_norm_rope(
        &self,
        q: &MxArray,
        k: &MxArray,
        position: i32,
    ) -> Option<(MxArray, MxArray)> {
        static METAL: OnceLock<bool> = OnceLock::new();
        let metal = *METAL.get_or_init(|| unsafe { mlx_sys::mlx_metal_is_available() });
        if !metal || self.rope.traditional {
            return None;
        }
        let (q_norm, k_norm) = self.q_norm.as_ref().zip(self.k_norm.as_ref())?;
        if q_norm.eps_f32() != k_norm.eps_f32() {
            return None;
        }
        let batch = q.shape_at(0).ok()?;
        let offsets = MxArray::from_int32(&vec![position; batch as usize], &[batch]).ok()?;
        let mut out_q = std::ptr::null_mut();
        let mut out_k = std::ptr::null_mut();
        // SAFETY: every handle is a live array for the call; the outputs are
        // owned handles or stay null when the call reports false.
        let ok = unsafe {
            mlx_sys::mlx_qk_norm_rope(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                q_norm.weight().as_raw_ptr(),
                k_norm.weight().as_raw_ptr(),
                offsets.as_raw_ptr(),
                q_norm.eps_f32(),
                self.rope.base,
                self.rope.scale,
                self.rope.dims,
                &mut out_q,
                &mut out_k,
            )
        };
        if !ok {
            return None;
        }
        Some((
            MxArray::from_handle(out_q, "qk_norm_rope:q").ok()?,
            MxArray::from_handle(out_k, "qk_norm_rope:k").ok()?,
        ))
    }

    pub fn forward(&self, x: &MxArray, state: &mut AttentionState) -> Result<MxArray> {
        let c = &self.config;
        let shape = x.shape()?;
        let (b, n) = (shape[0], shape[1]);
        let h = self.norm1.forward(x)?;
        let q = self
            .q
            .forward(&h)?
            .reshape(&[b, n, c.num_attention_heads, c.head_dim])?;
        let k = self
            .k
            .forward(&h)?
            .reshape(&[b, n, c.num_key_value_heads, c.head_dim])?;
        let v = self
            .v
            .forward(&h)?
            .reshape(&[b, n, c.num_key_value_heads, c.head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        // TTS uses identical temporal/height/width positions; interleaved MRoPE
        // therefore equals ordinary nontraditional RoPE for the three identical position axes.
        let (q, k) = match self.fused_qk_norm_rope(&q, &k, state.position as i32) {
            Some(qk) => {
                // Output cannot distinguish the branches (bit-identical), so
                // tests count engagements instead.
                #[cfg(test)]
                tests::FUSED_QK_NORM_ROPE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                qk
            }
            None => {
                let q = match &self.q_norm {
                    Some(norm) => norm.forward(&q)?,
                    None => q,
                };
                let k = match &self.k_norm {
                    Some(norm) => norm.forward(&k)?,
                    None => k,
                };
                (
                    self.rope.forward(
                        &q.transpose(Some(&[0, 2, 1, 3]))?,
                        Some(state.position as i32),
                    )?,
                    self.rope.forward(
                        &k.transpose(Some(&[0, 2, 1, 3]))?,
                        Some(state.position as i32),
                    )?,
                )
            }
        };
        let previous = state.kv.get_offset() as i64;
        // One KvStoreRows dispatch instead of two slice_assigns; bytes are
        // identical to update_and_fetch, which is also the fallback when the
        // fused store declines (no Metal). Batching across layers is not
        // possible: layer i's rows are produced by layer i-1's output.
        state
            .kv
            .store_block(&KvBlock::Bf16 { keys: k, values: v })?;
        let offset = i64::from(state.kv.get_offset());
        let k = state
            .kv
            .keys_ref()
            .ok_or_else(|| Error::from_reason("TTS KV keys missing after store"))?
            .slice_axis(2, 0, offset)?;
        let v = state
            .kv
            .values_ref()
            .ok_or_else(|| Error::from_reason("TTS KV values missing after store"))?
            .slice_axis(2, 0, offset)?;
        let total = previous + n;
        let scale = (c.head_dim as f64).powf(-0.5);
        let h = if let Some(window) = self.attention_window {
            // Causal-and-windowed masking is not expressible as an SDPA mask
            // mode, so windowed layers keep the explicit [n, total] mask.
            let mut data = vec![0f32; (n * total) as usize];
            for row in 0..n {
                for col in 0..total {
                    if col > previous + row || col + window as i64 <= previous + row {
                        data[(row * total + col) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = MxArray::from_float32(&data, &[n, total])?.astype(x.dtype()?)?;
            scaled_dot_product_attention(&q, &k, &v, scale, Some(&mask))?
        } else if n > 1 {
            // Purely causal prefill: the fused causal kernel aligns the last
            // n query rows with the kv tail, matching `previous + row` offsets.
            scaled_dot_product_attention_causal(&q, &k, &v, scale)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, scale, None)?
        }
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[b, n, c.num_attention_heads * c.head_dim])?;
        let mut h = self.o.forward(&h)?;
        if let Some(scale) = &self.attn_scale {
            h = h.mul(scale)?;
        }
        let x = x.add(&h)?;
        let h = self.norm2.forward(&x)?;
        let h = match &self.gate {
            Some(gate) => Activations::swiglu_compiled(&gate.forward(&h)?, &self.up.forward(&h)?)?,
            // Mimi's hidden_act="gelu" uses erf GELU (Transformers 4.57.3),
            // unlike the tanh approximation used by some other model families.
            None => Activations::gelu_exact(&self.up.forward(&h)?)?,
        };
        let mut h = self.down.forward(&h)?;
        if let Some(scale) = &self.mlp_scale {
            h = h.mul(scale)?;
        }
        state.position += n as usize;
        if let Some(window) = self.attention_window {
            state.kv.retain_recent(window.saturating_sub(1))?;
        }
        x.add(&h)
    }
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    norm: Option<RMSNorm>,
    pub dtype: DType,
}
impl Decoder {
    pub fn load(
        w: &Weights,
        prefix: &str,
        c: &TransformerConfig,
        qk: bool,
        scales: bool,
    ) -> Result<Self> {
        let layers = (0..c.num_hidden_layers)
            .map(|i| DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), c, qk, scales))
            .collect::<Result<_>>()?;
        let norm = w.rms(&format!("{prefix}.norm"), c.rms_norm_eps)?;
        let dtype = norm.get_weight().dtype()?;
        Ok(Self {
            layers,
            norm: Some(norm),
            dtype,
        })
    }
    pub fn load_encoder(w: &Weights, prefix: &str, c: &TransformerConfig) -> Result<Self> {
        c.validate()?;
        if c.hidden_act != "gelu" {
            return Err(Error::from_reason("Unsupported TTS encoder activation"));
        }
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                let a = format!("{p}.self_attn");
                Ok(DecoderLayer {
                    q: w.linear(&format!("{a}.q_proj"))?,
                    k: w.linear(&format!("{a}.k_proj"))?,
                    v: w.linear(&format!("{a}.v_proj"))?,
                    o: w.linear(&format!("{a}.o_proj"))?,
                    q_norm: None,
                    k_norm: None,
                    norm1: Norm::Layer(w.norm(&format!("{p}.input_layernorm"), c.rms_norm_eps)?),
                    norm2: Norm::Layer(
                        w.norm(&format!("{p}.post_attention_layernorm"), c.rms_norm_eps)?,
                    ),
                    gate: None,
                    up: w.linear(&format!("{p}.mlp.fc1"))?,
                    down: w.linear(&format!("{p}.mlp.fc2"))?,
                    attn_scale: Some(w.get(&format!("{p}.self_attn_layer_scale.scale"))?),
                    mlp_scale: Some(w.get(&format!("{p}.mlp_layer_scale.scale"))?),
                    config: c.clone(),
                    // Match Qwen's offline encoder and transformers 4.57.3
                    // Mimi SDPA/eager: its explicit causal mask does not apply
                    // config.sliding_window (unlike its FlashAttention path).
                    attention_window: None,
                    rope: RoPE::new(c.head_dim as i32, Some(false), Some(c.rope_theta), None),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            norm: None,
            dtype: w
                .get(&format!("{prefix}.layers.0.self_attn.q_proj.weight"))?
                .dtype()?,
        })
    }
    pub fn state(&self) -> Vec<AttentionState> {
        (0..self.layers.len())
            .map(|_| AttentionState::default())
            .collect()
    }
    pub fn forward(&self, input: &MxArray, states: &mut [AttentionState]) -> Result<MxArray> {
        let mut x = input.astype(self.dtype)?;
        for (layer, state) in self.layers.iter().zip(states) {
            x = layer.forward(&x, state)?;
        }
        match &self.norm {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Engagements of the fused q/k norm+rope branch in `forward`; the two
    /// paths produce identical bytes, so only a counter can observe which ran.
    pub(super) static FUSED_QK_NORM_ROPE_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn qwen3_tts_encoder_mlp_matches_exact_gelu() {
        let mut config: TransformerConfig = serde_json::from_value(
            serde_json::from_str::<serde_json::Value>(include_str!("fixtures/codec.json")).unwrap()
                ["encoder_config"]
                .clone(),
        )
        .unwrap();
        config.hidden_size = 2;
        config.intermediate_size = 2;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = 2;
        let zeros = MxArray::from_float32(&[0.; 4], &[2, 2]).unwrap();
        let identity = MxArray::from_float32(&[1., 0., 0., 1.], &[2, 2]).unwrap();
        let linear = |weight: &MxArray| Linear::from_weights(weight, None).unwrap();
        let layer = DecoderLayer {
            q: linear(&zeros),
            k: linear(&zeros),
            v: linear(&zeros),
            o: linear(&zeros),
            q_norm: None,
            k_norm: None,
            norm1: Norm::Layer(LayerNorm::new(2, Some(1e-6)).unwrap()),
            norm2: Norm::Layer(LayerNorm::new(2, Some(1e-6)).unwrap()),
            gate: None,
            up: linear(&identity),
            down: linear(&identity),
            attn_scale: None,
            mlp_scale: None,
            config,
            attention_window: None,
            rope: RoPE::new(2, Some(false), Some(10000.), None),
        };
        let input = MxArray::from_float32(&[-1., 1.], &[1, 1, 2]).unwrap();
        let output = layer
            .forward(&input, &mut AttentionState::default())
            .unwrap()
            .to_float32()
            .unwrap();
        // x + GELU(LayerNorm(x)); scalar erf reference with eps=1e-6.
        // The tanh approximation differs by ~1.53e-4 at both positions.
        for (actual, expected) in output.iter().zip([-1.158655295589131, 1.8413442044112442]) {
            assert!((f64::from(*actual) - expected).abs() < 2e-6);
        }
    }

    fn tiny_config(head_dim: i64) -> TransformerConfig {
        TransformerConfig {
            hidden_size: head_dim,
            intermediate_size: head_dim,
            num_hidden_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 32768,
            attention_bias: false,
            sliding_window: None,
        }
    }

    /// Layer whose attention path is exercised directly; the projection
    /// linears are placeholders never invoked by the fused helper.
    fn qk_norm_layer(c: &TransformerConfig) -> Result<DecoderLayer> {
        let zeros = MxArray::from_float32(&[0.; 4], &[2, 2])?;
        let linear = |weight: &MxArray| Linear::from_weights(weight, None).unwrap();
        Ok(DecoderLayer {
            q: linear(&zeros),
            k: linear(&zeros),
            v: linear(&zeros),
            o: linear(&zeros),
            q_norm: Some(RMSNorm::new(c.head_dim as u32, Some(c.rms_norm_eps))?),
            k_norm: Some(RMSNorm::new(c.head_dim as u32, Some(c.rms_norm_eps))?),
            norm1: Norm::Rms(RMSNorm::new(c.hidden_size as u32, Some(c.rms_norm_eps))?),
            norm2: Norm::Rms(RMSNorm::new(c.hidden_size as u32, Some(c.rms_norm_eps))?),
            gate: None,
            up: linear(&zeros),
            down: linear(&zeros),
            attn_scale: None,
            mlp_scale: None,
            config: c.clone(),
            attention_window: None,
            rope: RoPE::new(c.head_dim as i32, Some(false), Some(c.rope_theta), None),
        })
    }

    /// The fused `mlx_qk_norm_rope` dispatch must equal the op chain
    /// `rms_norm -> transpose -> rope(offset)` bit-for-bit; `forward` falls
    /// back to the chain silently on any divergence-prone contract miss, so a
    /// mismatch here would ship unnoticed. Talker/predictor geometry
    /// (16 q / 8 kv heads, D=128, theta=1e6) at prefill and decode lengths.
    #[test]
    fn qk_norm_rope_fused_matches_op_chain() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let c = tiny_config(128);
        let mut layer = qk_norm_layer(&c)?;
        layer
            .q_norm
            .as_mut()
            .unwrap()
            .set_weight(&MxArray::random_normal(
                &[c.head_dim],
                1.0,
                0.25,
                Some(DType::BFloat16),
            )?)?;
        layer
            .k_norm
            .as_mut()
            .unwrap()
            .set_weight(&MxArray::random_normal(
                &[c.head_dim],
                1.0,
                0.25,
                Some(DType::BFloat16),
            )?)?;
        for t in [1i64, 2, 4, 7] {
            for position in [0i32, 1, 7, 513, 32_480] {
                let q = MxArray::random_normal(
                    &[1, t, c.num_attention_heads, c.head_dim],
                    0.0,
                    1.0,
                    Some(DType::BFloat16),
                )?;
                let k = MxArray::random_normal(
                    &[1, t, c.num_key_value_heads, c.head_dim],
                    0.0,
                    1.0,
                    Some(DType::BFloat16),
                )?;
                let (fq, fk) = layer
                    .fused_qk_norm_rope(&q, &k, position)
                    .expect("fused qk norm+rope must take this contract");
                let rq = layer.rope.forward(
                    &layer
                        .q_norm
                        .as_ref()
                        .unwrap()
                        .forward(&q)?
                        .transpose(Some(&[0, 2, 1, 3]))?,
                    Some(position),
                )?;
                let rk = layer.rope.forward(
                    &layer
                        .k_norm
                        .as_ref()
                        .unwrap()
                        .forward(&k)?
                        .transpose(Some(&[0, 2, 1, 3]))?,
                    Some(position),
                )?;
                for (name, fused, reference) in [("q", &fq, &rq), ("k", &fk, &rk)] {
                    assert_eq!(fused.dtype()?, DType::BFloat16);
                    assert_eq!(fused.shape()?.as_ref(), reference.shape()?.as_ref());
                    let a = fused.astype(DType::Float32)?.to_float32()?;
                    let b = reference.astype(DType::Float32)?.to_float32()?;
                    let mismatches = a
                        .as_ref()
                        .iter()
                        .zip(b.as_ref().iter())
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    assert_eq!(
                        mismatches,
                        0,
                        "{name}: t={t} position={position}: {mismatches} of {} elements differ",
                        a.len()
                    );
                }
            }
        }
        Ok(())
    }

    /// The fused branch is selected by `forward`, not just by the helper:
    /// a gate regression (Metal probe, rope flag, eps mismatch) that silently
    /// restores the op chain must fail this test even though outputs stay
    /// bit-identical. Talker geometry (16 q / 8 kv heads, D=128).
    #[test]
    #[cfg(target_os = "macos")]
    fn forward_engages_fused_qk_norm_rope_at_model_geometry() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let c = tiny_config(128);
        let mut layer = qk_norm_layer(&c)?;
        let rand =
            |shape: &[i64]| MxArray::random_normal(shape, 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        layer.q = Linear::from_weights(
            &rand(&[c.num_attention_heads * c.head_dim, c.hidden_size]),
            None,
        )?;
        layer.k = Linear::from_weights(
            &rand(&[c.num_key_value_heads * c.head_dim, c.hidden_size]),
            None,
        )?;
        layer.v = Linear::from_weights(
            &rand(&[c.num_key_value_heads * c.head_dim, c.hidden_size]),
            None,
        )?;
        layer.o = Linear::from_weights(
            &rand(&[c.hidden_size, c.num_attention_heads * c.head_dim]),
            None,
        )?;
        layer.up = Linear::from_weights(&rand(&[c.intermediate_size, c.hidden_size]), None)?;
        layer.down = Linear::from_weights(&rand(&[c.hidden_size, c.intermediate_size]), None)?;
        let before = FUSED_QK_NORM_ROPE_CALLS.load(Ordering::Relaxed);
        let mut state = AttentionState::default();
        layer.forward(&rand(&[1, 7, c.hidden_size]), &mut state)?; // prefill
        layer.forward(&rand(&[1, 1, c.hidden_size]), &mut state)?; // decode row
        assert_eq!(
            FUSED_QK_NORM_ROPE_CALLS.load(Ordering::Relaxed) - before,
            2,
            "forward must take the fused qk norm+rope branch for this contract"
        );
        Ok(())
    }

    /// The causal SDPA fast path must reproduce the explicit `[n, total]`
    /// causal mask `forward` used to build for every prefill, including the
    /// kv-longer-than-q continuation shape (codec/text prefix resident in
    /// cache). Covers both production geometries: talker/predictor at D=128
    /// and the codec encoder at D=64. The masked and causal calls dispatch
    /// different kernel families, so bf16 outputs may differ by rounding;
    /// compare in f32 against a bf16-scaled bound (one ulp near |x|=4 is
    /// 0.03125) instead of an absolute f32 epsilon a single ulp exceeds.
    #[test]
    #[cfg(target_os = "macos")]
    fn causal_sdpa_matches_explicit_mask() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        for (hq, hk, d) in [(16i64, 8i64, 128i64), (8, 8, 64)] {
            let scale = (d as f64).powf(-0.5);
            for (previous, n) in [(0i64, 7), (33, 5)] {
                let total = previous + n;
                let q = MxArray::random_normal(&[1, hq, n, d], 0.0, 1.0, Some(DType::BFloat16))?;
                let k =
                    MxArray::random_normal(&[1, hk, total, d], 0.0, 1.0, Some(DType::BFloat16))?;
                let v =
                    MxArray::random_normal(&[1, hk, total, d], 0.0, 1.0, Some(DType::BFloat16))?;
                let mut data = vec![0f32; (n * total) as usize];
                for row in 0..n {
                    for col in 0..total {
                        if col > previous + row {
                            data[(row * total + col) as usize] = f32::NEG_INFINITY;
                        }
                    }
                }
                let mask = MxArray::from_float32(&data, &[n, total])?.astype(DType::BFloat16)?;
                let masked =
                    scaled_dot_product_attention(&q, &k, &v, scale, Some(&mask))?.to_float32()?;
                let causal =
                    scaled_dot_product_attention_causal(&q, &k, &v, scale)?.to_float32()?;
                let worst = masked
                    .as_ref()
                    .iter()
                    .zip(causal.as_ref().iter())
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 0.03125,
                    "hq={hq} d={d} previous={previous} n={n}: causal SDPA diverges from explicit mask by {worst}"
                );
            }
        }
        Ok(())
    }

    fn bits_equal(a: &MxArray, b: &MxArray) -> Result<bool> {
        if a.shape()?.as_ref() != b.shape()?.as_ref() {
            return Ok(false);
        }
        let a = a.astype(DType::Float32)?.to_float32()?;
        let b = b.astype(DType::Float32)?.to_float32()?;
        Ok(a.as_ref()
            .iter()
            .zip(b.as_ref().iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()))
    }

    /// `store_block` replaced `update_and_fetch` at the layer's KV append;
    /// both are documented byte-identical and this replays the TTS decode
    /// shapes through both caches: 1-row talker appends, the predictor's
    /// `reset_keep_capacity` refill cycle, and the codec's `retain_recent`
    /// trim (the [B,T,H,D]-transposed `v` view matches the forward's). The
    /// `kv_store_rows` dispatch count additionally proves the fused store
    /// engaged for these geometries — on a contract miss `store_block` falls
    /// back to the same `update_and_fetch` it is compared against, which
    /// would otherwise pass green while shipping the port as dead code.
    #[test]
    #[cfg(target_os = "macos")]
    fn kv_store_block_decode_matches_slice_update() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        fn block(b: i64, h: i64, d: i64, t: i64) -> Result<(MxArray, MxArray)> {
            let keys = MxArray::random_normal(&[b, h, t, d], 0.0, 1.0, Some(DType::BFloat16))?;
            let values = MxArray::random_normal(&[b, t, h, d], 0.0, 1.0, Some(DType::BFloat16))?
                .transpose(Some(&[0, 2, 1, 3]))?;
            Ok((keys, values))
        }
        fn append(sliced: &mut KVCache, stored: &mut KVCache, t: i64) -> Result<()> {
            let (keys, values) = block(1, 8, 128, t)?;
            sliced.update_and_fetch(&keys, &values)?;
            stored.store_block(&KvBlock::Bf16 { keys, values })?;
            Ok(())
        }
        let mut sliced = KVCache::new();
        let mut stored = KVCache::new();
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        append(&mut sliced, &mut stored, 4)?; // prefill
        for _ in 0..3 {
            append(&mut sliced, &mut stored, 1)?; // talker/codec decode rows
        }
        // Predictor: same cache rewritten from offset 0 each frame.
        sliced.reset_keep_capacity();
        stored.reset_keep_capacity();
        for _ in 0..2 {
            append(&mut sliced, &mut stored, 15)?;
            sliced.reset_keep_capacity();
            stored.reset_keep_capacity();
        }
        // Codec: rows beyond the window are dropped between appends.
        for _ in 0..3 {
            append(&mut sliced, &mut stored, 1)?;
            sliced.retain_recent(6)?;
            stored.retain_recent(6)?;
        }
        // Store evaluations are lazy; eval both buffers before counting.
        let keys = stored.keys_ref().unwrap().clone();
        let values = stored.values_ref().unwrap().clone();
        MxArray::eval_arrays(&[&keys, &values])?;
        let stores = unsafe { mlx_sys::mlx_test_kquant_family_count(c"kv_store_rows".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        assert_eq!(stores, 9, "every append must dispatch one KvStoreRows");
        assert_eq!(sliced.get_offset(), stored.get_offset());
        assert_eq!(sliced.capacity()?, stored.capacity()?);
        for (name, a, b) in [
            ("keys", sliced.keys_ref(), stored.keys_ref()),
            ("values", sliced.values_ref(), stored.values_ref()),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => assert!(bits_equal(a, b)?, "{name} buffers differ"),
                _ => panic!("{name} buffer presence differs"),
            }
        }
        Ok(())
    }

    /// `generate` reserves the talker KV bound before the loop; reservation
    /// changes only spare capacity, so a reserved cache must produce
    /// bit-identical layer outputs whether primed empty or live.
    #[test]
    fn reserve_does_not_change_talker_decode() -> Result<()> {
        let mut c = tiny_config(4);
        c.hidden_size = 16;
        c.intermediate_size = 16;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 2;
        let rand =
            |shape: &[i64]| MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32)).unwrap();
        let layer = DecoderLayer {
            q: Linear::from_weights(&rand(&[8, 16]), None)?,
            k: Linear::from_weights(&rand(&[8, 16]), None)?,
            v: Linear::from_weights(&rand(&[8, 16]), None)?,
            o: Linear::from_weights(&rand(&[16, 8]), None)?,
            q_norm: Some(RMSNorm::new(4, Some(1e-6))?),
            k_norm: Some(RMSNorm::new(4, Some(1e-6))?),
            norm1: Norm::Rms(RMSNorm::new(16, Some(1e-6))?),
            norm2: Norm::Rms(RMSNorm::new(16, Some(1e-6))?),
            gate: Some(Linear::from_weights(&rand(&[16, 16]), None)?),
            up: Linear::from_weights(&rand(&[16, 16]), None)?,
            down: Linear::from_weights(&rand(&[16, 16]), None)?,
            attn_scale: None,
            mlp_scale: None,
            config: c,
            attention_window: None,
            rope: RoPE::new(4, Some(false), Some(1e6), None),
        };
        let prompt = rand(&[1, 5, 16]);
        let steps: Vec<MxArray> = (0..8).map(|_| rand(&[1, 1, 16])).collect();
        let bound = 5 + steps.len() as i64;

        let mut plain = AttentionState::default();
        // Small growth step so reserving past the first allocation exercises
        // the live-buffer grow, the path a prefix-cache hit takes.
        let mut reserved_empty = AttentionState {
            kv: KVCache::with_growth_step(4)?,
            position: 0,
        };
        reserved_empty.kv.reserve(bound)?;
        let mut reserved_live = AttentionState {
            kv: KVCache::with_growth_step(4)?,
            position: 0,
        };

        let reference = layer.forward(&prompt, &mut plain)?;
        assert!(bits_equal(
            &layer.forward(&prompt, &mut reserved_empty)?,
            &reference
        )?);
        assert!(bits_equal(
            &layer.forward(&prompt, &mut reserved_live)?,
            &reference
        )?);
        reserved_live.kv.reserve(bound)?;
        assert!(reserved_empty.kv.capacity()? >= bound);
        assert!(reserved_live.kv.capacity()? >= bound);

        for (i, step) in steps.iter().enumerate() {
            let expected = layer.forward(step, &mut plain)?;
            for (name, state) in [("empty", &mut reserved_empty), ("live", &mut reserved_live)] {
                let actual = layer.forward(step, state)?;
                assert!(bits_equal(&actual, &expected)?, "{name} step {i}");
                assert_eq!(
                    state.kv.get_offset(),
                    plain.kv.get_offset(),
                    "{name} step {i}"
                );
            }
        }
        Ok(())
    }
}
