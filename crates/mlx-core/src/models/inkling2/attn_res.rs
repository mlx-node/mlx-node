use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::nn::Activations;
use crate::nn::normalization::{rms_norm_scaled, rms_norm_unscaled};

/// Hardcoded AttnRes / loop-controller RMSNorm epsilon (`1e-6`, matches vLLM and llama.cpp).
pub const ATTN_RES_EPS: f32 = 1e-6;

/// Block AttnRes mix over `sources = [...blocks, partial]` (or a cross-pass `bank`)
/// with pre-scaled queries `scaled_queries` (`[Q, D]` in FP32, already multiplied by `D^-0.5`).
///
/// Exact math (`normalize_values = true`, `eps = 1e-6`):
/// 1. Stack all `N` sources into `[N, T, D]` in FP32.
/// 2. Normalize every source independently: `v_n = s_n * rsqrt(mean(s_n^2) + 1e-6)`.
/// 3. Scores: `logits[q, t, n] = sum_d scaled_queries[q, d] * v_n[t, d]`.
/// 4. Weights: `p = softmax(logits, axis=N)`.
/// 5. Output: `y[q, t, d] = sum_n p[q, t, n] * v_n[t, d]` in **FP32** (`[Q, T, D]`).
pub fn block_attn_res_mix(sources: &[&MxArray], scaled_queries: &MxArray) -> Result<MxArray> {
    if sources.is_empty() {
        return Err(Error::from_reason("block_attn_res_mix: empty source list"));
    }
    let q_shape = scaled_queries.shape()?;
    if q_shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "block_attn_res_mix: expected 2D queries [Q, D], got {:?}",
            q_shape.as_ref()
        )));
    }
    let num_q = q_shape[0];
    let dim = q_shape[1];

    let q_f32 = if scaled_queries.dtype()? == DType::Float32 {
        scaled_queries.clone()
    } else {
        scaled_queries.astype(DType::Float32)?
    };

    // Fast path for a single source: softmax([s]) == 1.0, output is normalized source repeated Q times.
    if sources.len() == 1 {
        let s0_f32 = sources[0].astype(DType::Float32)?;
        let v0 = rms_norm_unscaled(&s0_f32, ATTN_RES_EPS)?;
        let t = v0.shape()?[0];
        return v0.expand_dims(0)?.broadcast_to(&[num_q, t, dim]);
    }

    let mut f32_sources = Vec::with_capacity(sources.len());
    for s in sources {
        f32_sources.push(s.astype(DType::Float32)?);
    }
    let refs: Vec<&MxArray> = f32_sources.iter().collect();
    // `[N, T, D]`
    let stacked = MxArray::stack(refs, Some(0))?;

    // Weightless RMSNorm along last dim D in FP32 -> `[N, T, D]`
    let normed = rms_norm_unscaled(&stacked, ATTN_RES_EPS)?;

    // Transpose to `[T, N, D]` so matmul with `q_f32.T` (`[D, Q]`) yields `[T, N, Q]`
    let v_tnd = normed.transpose(Some(&[1, 0, 2]))?;
    let q_t = q_f32.transpose(None)?; // `[D, Q]`
    let scores_tnq = v_tnd.matmul(&q_t)?; // `[T, N, Q]`
    let scores_tqn = scores_tnq.transpose(Some(&[0, 2, 1]))?; // `[T, Q, N]`
    let probs_tqn = Activations::softmax_precise(&scores_tqn, Some(-1))?; // `[T, Q, N]`

    // Weighted sum: `[T, Q, N] @ [T, N, D]` -> `[T, Q, D]` -> `[Q, T, D]`
    let mixed_tqd = probs_tqn.matmul(&v_tnd)?;
    mixed_tqd.transpose(Some(&[1, 0, 2]))
}

/// Normalize one `[T, D]` AttnRes source in FP32 and reshape as `[T, 1, D]` so
/// completed blocks and cross-pass banks can be concatenated directly along axis 1
/// (`[T, N, D]`) with zero repeated casts, norms, or transposes.
pub fn normalize_attn_res_source(source: &MxArray) -> Result<MxArray> {
    let shape = source.shape()?;
    if shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "normalize_attn_res_source: expected 2D [T, D], got {:?}",
            shape.as_ref()
        )));
    }
    let t = shape[0];
    let d = shape[1];
    let s_f32 = if source.dtype()? == DType::Float32 {
        source.clone()
    } else {
        source.astype(DType::Float32)?
    };
    rms_norm_unscaled(&s_f32, ATTN_RES_EPS)?.reshape(&[t, 1, d])
}

/// Concatenate pre-normalized `[T, 1, D]` FP32 sources (plus optional trailing current
/// `[T, 1, D]` source) along axis 1 into `[T, N, D]` FP32.
pub fn stack_normed_sources_tnd(
    normed_blocks: &[MxArray],
    extra: Option<&MxArray>,
) -> Result<MxArray> {
    let total = normed_blocks.len() + usize::from(extra.is_some());
    if total == 0 {
        return Err(Error::from_reason(
            "stack_normed_sources_tnd: empty source list",
        ));
    }
    if total == 1 {
        if let Some(first) = normed_blocks.first() {
            return Ok(first.clone());
        }
        if let Some(e) = extra {
            return Ok(e.clone());
        }
    }
    let mut refs: Vec<&MxArray> = Vec::with_capacity(total);
    for b in normed_blocks {
        refs.push(b);
    }
    if let Some(e) = extra {
        refs.push(e);
    }
    MxArray::concatenate_many(refs, Some(1))
}

/// Block AttnRes mix over pre-normalized `v_tnd` (`[T, N, D]` FP32) using pre-transposed
/// `scaled_queries_t` (`[D, Q]` FP32).
///
/// Returns `[Q, T, D]` in FP32.
pub fn block_attn_res_mix_prenormed(
    v_tnd: &MxArray,
    scaled_queries_t: &MxArray,
) -> Result<MxArray> {
    let v_shape = v_tnd.shape()?;
    let q_shape = scaled_queries_t.shape()?;
    let t = v_shape[0];
    let n = v_shape[1];
    let d = v_shape[2];
    let num_q = q_shape[1];

    if n == 1 {
        let v0 = v_tnd.reshape(&[1, t, d])?;
        return if num_q == 1 {
            Ok(v0)
        } else {
            v0.broadcast_to(&[num_q, t, d])
        };
    }

    let scores_tnq = v_tnd.matmul(scaled_queries_t)?; // `[T, N, Q]`
    let scores_tqn = if num_q == 1 {
        scores_tnq.reshape(&[t, 1, n])?
    } else {
        scores_tnq.transpose(Some(&[0, 2, 1]))?
    };
    let probs_tqn = Activations::softmax_precise(&scores_tqn, Some(-1))?; // `[T, Q, N]`
    let mixed_tqd = probs_tqn.matmul(v_tnd)?; // `[T, Q, D]`
    if num_q == 1 {
        mixed_tqd.reshape(&[1, t, d])
    } else {
        mixed_tqd.transpose(Some(&[1, 0, 2]))
    }
}

/// Fast single-query (`Q = 1`) Block AttnRes mix over cached `normed_blocks` (`[T, 1, D]` each)
/// plus `current_norm_t1d` (`[T, 1, D]`), returning `[T, D]` FP32 directly with zero transposes
/// or slices!
pub fn block_attn_res_mix_single_q_prenormed(
    normed_blocks: &[MxArray],
    current_norm_t1d: &MxArray,
    scaled_query_t: &MxArray,
) -> Result<MxArray> {
    let shape = current_norm_t1d.shape()?;
    let t = shape[0];
    let d = shape[2];
    if normed_blocks.is_empty() {
        return current_norm_t1d.reshape(&[t, d]);
    }
    let v_tnd = stack_normed_sources_tnd(normed_blocks, Some(current_norm_t1d))?;
    let n = (normed_blocks.len() + 1) as i64;
    let scores_t1n = v_tnd.matmul(scaled_query_t)?.reshape(&[t, 1, n])?;
    let probs_t1n = Activations::softmax_precise(&scores_t1n, Some(-1))?;
    probs_t1n.matmul(&v_tnd)?.reshape(&[t, d])
}

/// Apply sublayer RMSNorm (`weight` `[D]`, `eps`) to a `[T, D]` FP32/BF16 tensor,
/// returning `out_dtype` (BF16).
pub fn apply_sublayer_rms_norm(
    x: &MxArray,
    weight: &MxArray,
    eps: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let x_f32 = if x.dtype()? == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    let w_f32 = if weight.dtype()? == DType::Float32 {
        weight.clone()
    } else {
        weight.astype(DType::Float32)?
    };
    let normed = rms_norm_scaled(&x_f32, &w_f32, eps)?;
    if out_dtype != DType::Float32 {
        normed.astype(out_dtype)
    } else {
        Ok(normed)
    }
}

/// Min/max token batch size `T` that uses the custom coalesced exact-FP32 LoopController Metal kernels.
#[inline]
pub(crate) fn fused_loop_ctrl_range() -> (i64, i64) {
    static RANGE: std::sync::OnceLock<(i64, i64)> = std::sync::OnceLock::new();
    *RANGE.get_or_init(|| {
        let min_t = std::env::var("MLX_INKLING2_LOOP_CTRL_MIN_T")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(1);
        let max_t = std::env::var("MLX_INKLING2_LOOP_CTRL_MAX_T")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(1024);
        (min_t, max_t)
    })
}

/// Resolved `MLX_INKLING2_LOOP_BF16` policy shared between `LoopController` loading and SSD
/// cache fingerprinting (`"bf16"`, `"f32"`, or `"auto"`).
#[inline]
pub(crate) fn loop_bf16_override_tag() -> &'static str {
    static MODE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    MODE.get_or_init(
        || match std::env::var("MLX_INKLING2_LOOP_BF16").as_deref() {
            Ok("1") | Ok("true") | Ok("bf16") => "bf16",
            Ok("0") | Ok("false") | Ok("f32") => "f32",
            _ => "auto",
        },
    )
}

/// Rank-64 low-rank loop controller for one stream (`q`/`attn`, `k`, `v`, or `mlp`).
///
/// Stores contiguous row-major `[rank, 2 * D]` (`down_proj_w`) and `[2 * D, rank]`
/// (`gain_forget_w`) FP32 weights for sequential 256B-aligned `float4` loads in the
/// fused Metal LoopController kernels, plus zero-copy transpose views (`down_proj_t` and
/// `gain_forget_proj_t`) for large-T prefill fallback!
pub struct LoopController {
    /// Row-major `[rank, 2 * D]` FP32 (`down_proj`, coalesced for `inkling2_loop_ctrl_down_f32`).
    pub down_proj_w: MxArray,
    /// Pre-transposed `[2 * D, rank]` FP32 (`down_proj_w.T`, zero-copy view).
    pub down_proj_t: MxArray,
    /// Row-major fused `[2 * D, rank]` FP32 (`concat([gain_proj, forget_proj], axis=0)`),
    /// sequential 256B `float4` layout for `inkling2_loop_ctrl_up_blend_f32`.
    pub gain_forget_w: MxArray,
    /// Pre-transposed fused `[rank, 2 * D]` FP32 (`gain_forget_w.T`, zero-copy view).
    pub gain_forget_proj_t: MxArray,
    pub dim: i64,
}

/// LoopController weight dtype: preserves the checkpoint's native precision (BF16 for original
/// SafeTensors where controllers are natively BF16 -> 2x less DRAM with zero precision loss;
/// FP32 for GGUF where controllers are natively FP32). Honors explicit `MLX_INKLING2_LOOP_BF16`
/// override when set (`1`/`true`/`bf16` or `0`/`false`/`f32`).
pub(crate) fn loop_controller_weight_dtype(source_dtype: DType) -> DType {
    match loop_bf16_override_tag() {
        "bf16" => DType::BFloat16,
        "f32" => DType::Float32,
        _ => {
            if source_dtype == DType::BFloat16 {
                DType::BFloat16
            } else {
                DType::Float32
            }
        }
    }
}

impl LoopController {
    pub fn new(down_proj: MxArray, gain_proj: MxArray, forget_proj: MxArray) -> Result<Self> {
        let dim = gain_proj.shape()?[0];
        let w_dtype = loop_controller_weight_dtype(down_proj.dtype()?);
        let down_proj_w = if down_proj.dtype()? == w_dtype {
            down_proj
        } else {
            down_proj.astype(w_dtype)?
        };
        let down_proj_t = down_proj_w.transpose(None)?;
        // Row-major `[2 * D, rank]` (`concat([gain, forget], axis=0)`): sequential coalesced
        // loads per output channel in `inkling2_loop_ctrl_up_blend_f32` AND zero-copy transpose
        // view `[rank, 2 * D]` for large-T prefill matmul (0 extra RAM!).
        let gain_w = if gain_proj.dtype()? == w_dtype {
            gain_proj
        } else {
            gain_proj.astype(w_dtype)?
        };
        let forget_w = if forget_proj.dtype()? == w_dtype {
            forget_proj
        } else {
            forget_proj.astype(w_dtype)?
        };
        let gain_forget_w = MxArray::concatenate(&gain_w, &forget_w, 0)?;
        let gain_forget_proj_t = gain_forget_w.transpose(None)?;
        Ok(Self {
            down_proj_w,
            down_proj_t,
            gain_forget_w,
            gain_forget_proj_t,
            dim,
        })
    }

    /// Compute the raw pre-`tanh` `(gain_raw, forget_raw)` projections `[T, D]` in FP32,
    /// optionally reusing already-normalized `d_hat` / `m_hat` (`[T, D]` FP32).
    pub fn compute_raw_gates_with_hats(
        &self,
        dest: &MxArray,
        msg: &MxArray,
        d_hat_opt: Option<&MxArray>,
        m_hat_opt: Option<&MxArray>,
    ) -> Result<(MxArray, MxArray)> {
        let d_hat_owned;
        let d_hat = match d_hat_opt {
            Some(dh) => dh,
            None => {
                let dest_f32 = if dest.dtype()? == DType::Float32 {
                    dest.clone()
                } else {
                    dest.astype(DType::Float32)?
                };
                d_hat_owned = rms_norm_unscaled(&dest_f32, ATTN_RES_EPS)?;
                &d_hat_owned
            }
        };
        let m_hat_owned;
        let m_hat = match m_hat_opt {
            Some(mh) => mh,
            None => {
                let msg_f32 = if msg.dtype()? == DType::Float32 {
                    msg.clone()
                } else {
                    msg.astype(DType::Float32)?
                };
                m_hat_owned = rms_norm_unscaled(&msg_f32, ATTN_RES_EPS)?;
                &m_hat_owned
            }
        };

        let t = d_hat.shape()?[0];
        let d = self.dim;
        let w_dtype = self.down_proj_t.dtype()?;
        let pair_f32 = MxArray::concatenate(d_hat, m_hat, -1)?;
        let pair = if pair_f32.dtype()? == w_dtype {
            pair_f32
        } else {
            pair_f32.astype(w_dtype)?
        };
        let h = pair.matmul(&self.down_proj_t)?;
        let gf = h.matmul(&self.gain_forget_proj_t)?;
        let gf_f32 = if gf.dtype()? == DType::Float32 {
            gf
        } else {
            gf.astype(DType::Float32)?
        };
        let gain_raw = gf_f32.slice(&[0, 0], &[t, d])?;
        let forget_raw = gf_f32.slice(&[0, d], &[t, 2 * d])?;
        Ok((gain_raw, forget_raw))
    }

    /// Compute the raw pre-`tanh` `(gain_raw, forget_raw)` projections `[T, D]` in FP32.
    pub fn compute_raw_gates(&self, dest: &MxArray, msg: &MxArray) -> Result<(MxArray, MxArray)> {
        self.compute_raw_gates_with_hats(dest, msg, None, None)
    }

    /// Blend FP32 `dest` `[T, D]` with FP32 cross-pass `msg` `[T, D]`:
    /// 1. `pair = concat([rms_norm(dest, 1e-6), rms_norm(msg, 1e-6)], axis=-1)` (`[T, 2D]`)
    /// 2. `h = pair @ down.T` (`[T, rank]`)
    /// 3. `[gain, forget] = h @ [gain.T, forget.T]` (`[T, 2D]`, FP32)
    /// 4. `out = (1 - tanh(forget)) * dest + tanh(gain) * msg` (FP32)
    pub fn forward(&self, dest: &MxArray, msg: &MxArray) -> Result<MxArray> {
        self.forward_with_hats(dest, msg, None, None)
    }

    pub fn forward_with_hats(
        &self,
        dest: &MxArray,
        msg: &MxArray,
        d_hat_opt: Option<&MxArray>,
        m_hat_opt: Option<&MxArray>,
    ) -> Result<MxArray> {
        let dest_f32 = if dest.dtype()? == DType::Float32 {
            dest.clone()
        } else {
            dest.astype(DType::Float32)?
        };
        let msg_f32 = if msg.dtype()? == DType::Float32 {
            msg.clone()
        } else {
            msg.astype(DType::Float32)?
        };
        let (gain_raw, forget_raw) =
            self.compute_raw_gates_with_hats(&dest_f32, &msg_f32, d_hat_opt, m_hat_opt)?;
        let gain = gain_raw.tanh()?;
        let forget = forget_raw.tanh()?;

        // `(1 - tanh(forget)) * dest + tanh(gain) * msg` in FP32
        let kept = dest_f32.sub(&forget.mul(&dest_f32)?)?;
        let added = gain.mul(&msg_f32)?;
        kept.add(&added)
    }

    /// Fused Metal path: for `T <= 32` with FP32 weights, runs coalesced exact-FP32
    /// `inkling2_loop_ctrl_down` + `inkling2_loop_ctrl_up_blend_norm` (zero `[T, 2D]` DRAM
    /// intermediates and zero 64x64-padded FP32 `steel_matmul`!). Falls back to MLX matmul
    /// for large prefill (`T > 32`) or BF16 weights, and cleanly on non-Metal.
    pub fn forward_fused_rms_norm(
        &self,
        dest: &MxArray,
        msg: &MxArray,
        norm_weight: &MxArray,
        eps: f32,
        out_dtype: DType,
    ) -> Result<MxArray> {
        #[cfg(target_os = "macos")]
        {
            if unsafe { mlx_sys::mlx_metal_is_available() } {
                let t = msg.shape_at(0)?;
                let d = self.dim;
                let rank = self.down_proj_w.shape_at(0)?;
                let w_dtype = self.down_proj_w.dtype()?;
                let (min_t, max_t) = fused_loop_ctrl_range();
                if (w_dtype == DType::Float32 || w_dtype == DType::BFloat16)
                    && t >= min_t
                    && t <= max_t
                    && d % 4 == 0
                    && rank % 4 == 0
                {
                    return super::kernels::inkling2_loop_ctrl_fused_single(
                        dest,
                        msg,
                        &self.down_proj_w,
                        &self.gain_forget_w,
                        norm_weight,
                        ATTN_RES_EPS,
                        eps,
                        out_dtype,
                    );
                }
                let pair =
                    super::kernels::inkling2_pair_rms_norm(dest, msg, ATTN_RES_EPS, w_dtype)?;
                let h = pair.matmul(&self.down_proj_t)?;
                let gf = h.matmul(&self.gain_forget_proj_t)?;
                let gain_raw = gf.slice(&[0, 0], &[t, d])?;
                let forget_raw = gf.slice(&[0, d], &[t, 2 * d])?;
                return super::kernels::inkling2_loop_blend_rms_norm(
                    dest,
                    msg,
                    &gain_raw,
                    &forget_raw,
                    norm_weight,
                    eps,
                    out_dtype,
                );
            }
        }
        self.forward_fused_rms_norm_with_hats(dest, msg, None, None, norm_weight, eps, out_dtype)
    }

    /// Same as [`Self::forward_fused_rms_norm`], reusing optional pre-computed `d_hat` and `m_hat`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fused_rms_norm_with_hats(
        &self,
        dest: &MxArray,
        msg: &MxArray,
        d_hat_opt: Option<&MxArray>,
        m_hat_opt: Option<&MxArray>,
        norm_weight: &MxArray,
        eps: f32,
        out_dtype: DType,
    ) -> Result<MxArray> {
        let (gain_raw, forget_raw) =
            self.compute_raw_gates_with_hats(dest, msg, d_hat_opt, m_hat_opt)?;
        #[cfg(target_os = "macos")]
        {
            if unsafe { mlx_sys::mlx_metal_is_available() } {
                return super::kernels::inkling2_loop_blend_rms_norm(
                    dest,
                    msg,
                    &gain_raw,
                    &forget_raw,
                    norm_weight,
                    eps,
                    out_dtype,
                );
            }
        }
        let dest_f32 = if dest.dtype()? == DType::Float32 {
            dest.clone()
        } else {
            dest.astype(DType::Float32)?
        };
        let msg_f32 = if msg.dtype()? == DType::Float32 {
            msg.clone()
        } else {
            msg.astype(DType::Float32)?
        };
        let gain = gain_raw.tanh()?;
        let forget = forget_raw.tanh()?;
        let blended = dest_f32
            .sub(&forget.mul(&dest_f32)?)?
            .add(&gain.mul(&msg_f32)?)?;
        apply_sublayer_rms_norm(&blended, norm_weight, eps, out_dtype)
    }

    pub fn weight_refs(&self) -> [&MxArray; 2] {
        [&self.down_proj_w, &self.gain_forget_w]
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        let mut promoted = 0usize;
        let db = self.down_proj_w.promote_cuda_device()?;
        if db > 0 {
            self.down_proj_t = self.down_proj_w.transpose(None)?;
            promoted += db;
        }
        let gfb = self.gain_forget_w.promote_cuda_device()?;
        if gfb > 0 {
            self.gain_forget_proj_t = self.gain_forget_w.transpose(None)?;
            promoted += gfb;
        }
        Ok(promoted)
    }
}

/// Cross-pass message queries + 4 rank-64 controllers for one physical core layer (`6..17`).
///
/// Stream order matches both vLLM and mainline llama.cpp:
/// `0 = q/attn`, `1 = k`, `2 = v`, `3 = mlp`.
pub struct LoopLayerControllers {
    /// Pre-scaled (`* D^-0.5`) row-major FP32 message queries `[4, D]` in order `[q, k, v, mlp]`.
    pub msg_queries: MxArray,
    /// Pre-scaled (`* D^-0.5`) & pre-transposed FP32 message queries `[D, 4]` (flat fallback).
    pub msg_queries_t: MxArray,
    /// Batched row-major Q/K/V down projections `[3, rank, 2 * D]` FP32 (coalesced for fused Metal kernel).
    pub qkv_down_w: MxArray,
    /// Batched Q/K/V down projections `[3, 2 * D, rank]` FP32 (zero-copy view of `qkv_down_w`).
    pub qkv_down_proj_t: MxArray,
    /// Batched row-major Q/K/V gain+forget projections `[3, 2 * D, rank]` FP32 (sequential 256B `float4` layout).
    pub qkv_gain_forget_w: MxArray,
    /// Batched Q/K/V gain+forget projections `[3, rank, 2 * D]` FP32 (zero-copy view of `qkv_gain_forget_w`).
    pub qkv_gain_forget_proj_t: MxArray,
    pub q_ctrl: LoopController,
    pub k_ctrl: LoopController,
    pub v_ctrl: LoopController,
    pub mlp_ctrl: LoopController,
}

impl LoopLayerControllers {
    /// Construct from pre-scaled `[4, D]` `msg_queries` and the 4 stream controllers, packing
    /// `[q, k, v]` into contiguous batched `[3, rank, 2 * D]` / `[3, 2 * D, rank]` weights (0 extra RAM!).
    pub fn new(
        msg_queries: MxArray,
        q_ctrl: LoopController,
        k_ctrl: LoopController,
        v_ctrl: LoopController,
        mlp_ctrl: LoopController,
    ) -> Result<Self> {
        let msg_queries_t = msg_queries.transpose(None)?;
        let qkv_down_w = MxArray::stack(
            vec![
                &q_ctrl.down_proj_w,
                &k_ctrl.down_proj_w,
                &v_ctrl.down_proj_w,
            ],
            Some(0),
        )?;
        let qkv_down_proj_t = qkv_down_w.transpose(Some(&[0, 2, 1]))?;

        let qkv_gain_forget_w = MxArray::stack(
            vec![
                &q_ctrl.gain_forget_w,
                &k_ctrl.gain_forget_w,
                &v_ctrl.gain_forget_w,
            ],
            Some(0),
        )?;
        let qkv_gain_forget_proj_t = qkv_gain_forget_w.transpose(Some(&[0, 2, 1]))?;

        let dim = q_ctrl.dim;
        let down_shape = qkv_down_w.shape()?;
        let gf_shape = qkv_gain_forget_w.shape()?;
        let slice_stream = |idx: i64| -> Result<LoopController> {
            let down_w = qkv_down_w
                .slice(&[idx, 0, 0], &[idx + 1, down_shape[1], down_shape[2]])?
                .squeeze(Some(&[0]))?;
            let down_t = down_w.transpose(None)?;
            let gf_w = qkv_gain_forget_w
                .slice(&[idx, 0, 0], &[idx + 1, gf_shape[1], gf_shape[2]])?
                .squeeze(Some(&[0]))?;
            let gf_t = gf_w.transpose(None)?;
            Ok(LoopController {
                down_proj_w: down_w,
                down_proj_t: down_t,
                gain_forget_w: gf_w,
                gain_forget_proj_t: gf_t,
                dim,
            })
        };
        Ok(Self {
            msg_queries,
            msg_queries_t,
            q_ctrl: slice_stream(0)?,
            k_ctrl: slice_stream(1)?,
            v_ctrl: slice_stream(2)?,
            qkv_down_w,
            qkv_down_proj_t,
            qkv_gain_forget_w,
            qkv_gain_forget_proj_t,
            mlp_ctrl,
        })
    }

    /// Compute all 4 cross-pass messages `[4, T, D]` in FP32 over pre-normalized `bank_v_tnd`
    /// (`[T, N_bank, D]` FP32, flat fallback).
    pub fn read_messages_prenormed(&self, bank_v_tnd: &MxArray) -> Result<MxArray> {
        block_attn_res_mix_prenormed(bank_v_tnd, &self.msg_queries_t)
    }

    /// Batched fused Q/K/V loop controllers + sublayer RMSNorm -> `(x_q, x_k, x_v)`!
    ///
    /// - `dest`: `[T, D]` (SWA broadcast across Q/K/V) or `[3, T, D]` (Global) FP32
    /// - `msgs_qkv`: `[3, T, D]` FP32 (`msgs[0..3]`)
    /// - `norm_weight`: `[D]` (SWA broadcast) or `[3, D]` (Global) FP32
    pub fn forward_qkv_fused_rms_norm(
        &self,
        dest: &MxArray,
        msgs_qkv: &MxArray,
        norm_weight: &MxArray,
        eps: f32,
        out_dtype: DType,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let t = msgs_qkv.shape_at(1)?;
        let d = msgs_qkv.shape_at(2)?;
        #[cfg(target_os = "macos")]
        {
            if unsafe { mlx_sys::mlx_metal_is_available() } {
                let rank = self.qkv_down_w.shape_at(1)?;
                let w_dtype = self.qkv_down_w.dtype()?;
                let (min_t, max_t) = fused_loop_ctrl_range();
                if (w_dtype == DType::Float32 || w_dtype == DType::BFloat16)
                    && t >= min_t
                    && t <= max_t
                    && d % 4 == 0
                    && rank % 4 == 0
                {
                    return super::kernels::inkling2_loop_ctrl_fused_qkv(
                        dest,
                        msgs_qkv,
                        &self.qkv_down_w,
                        &self.qkv_gain_forget_w,
                        norm_weight,
                        ATTN_RES_EPS,
                        eps,
                        out_dtype,
                    );
                }
                let pair_3 =
                    super::kernels::inkling2_pair_rms_norm(dest, msgs_qkv, ATTN_RES_EPS, w_dtype)?;
                let h_3 = pair_3.matmul(&self.qkv_down_proj_t)?;
                let gf_3 = h_3.matmul(&self.qkv_gain_forget_proj_t)?;
                let gain_3 = gf_3.slice(&[0, 0, 0], &[3, t, d])?;
                let forget_3 = gf_3.slice(&[0, 0, d], &[3, t, 2 * d])?;
                let out_3 = super::kernels::inkling2_loop_blend_rms_norm(
                    dest,
                    msgs_qkv,
                    &gain_3,
                    &forget_3,
                    norm_weight,
                    eps,
                    out_dtype,
                )?;
                let x_q = out_3.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?;
                let x_k = out_3.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?;
                let x_v = out_3.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?;
                return Ok((x_q, x_k, x_v));
            }
        }
        let msg_q = msgs_qkv
            .slice(&[0, 0, 0], &[1, t, d])?
            .squeeze(Some(&[0]))?;
        let msg_k = msgs_qkv
            .slice(&[1, 0, 0], &[2, t, d])?
            .squeeze(Some(&[0]))?;
        let msg_v = msgs_qkv
            .slice(&[2, 0, 0], &[3, t, d])?
            .squeeze(Some(&[0]))?;
        let (dq, dk, dv) = if dest.ndim()? == 3 {
            (
                dest.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?,
                dest.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?,
                dest.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?,
            )
        } else {
            (dest.clone(), dest.clone(), dest.clone())
        };
        let (wq, wk, wv) = if norm_weight.ndim()? == 2 {
            (
                norm_weight.slice(&[0, 0], &[1, d])?.squeeze(Some(&[0]))?,
                norm_weight.slice(&[1, 0], &[2, d])?.squeeze(Some(&[0]))?,
                norm_weight.slice(&[2, 0], &[3, d])?.squeeze(Some(&[0]))?,
            )
        } else {
            (
                norm_weight.clone(),
                norm_weight.clone(),
                norm_weight.clone(),
            )
        };
        Ok((
            self.q_ctrl
                .forward_fused_rms_norm(&dq, &msg_q, &wq, eps, out_dtype)?,
            self.k_ctrl
                .forward_fused_rms_norm(&dk, &msg_k, &wk, eps, out_dtype)?,
            self.v_ctrl
                .forward_fused_rms_norm(&dv, &msg_v, &wv, eps, out_dtype)?,
        ))
    }

    pub fn weight_refs(&self) -> Vec<&MxArray> {
        let mut refs = Vec::with_capacity(6);
        refs.push(&self.msg_queries);
        refs.push(&self.msg_queries_t);
        refs.push(&self.qkv_down_w);
        refs.push(&self.qkv_gain_forget_w);
        refs.extend_from_slice(&self.mlp_ctrl.weight_refs());
        refs
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = 0usize;
        let mqb = self.msg_queries.promote_cuda_device()?;
        if mqb > 0 {
            self.msg_queries_t = self.msg_queries.transpose(None)?;
            promoted += mqb;
        }
        let db = self.qkv_down_w.promote_cuda_device()?;
        if db > 0 {
            self.qkv_down_proj_t = self.qkv_down_w.transpose(Some(&[0, 2, 1]))?;
            promoted += db;
        }
        let gfb = self.qkv_gain_forget_w.promote_cuda_device()?;
        if gfb > 0 {
            self.qkv_gain_forget_proj_t = self.qkv_gain_forget_w.transpose(Some(&[0, 2, 1]))?;
            promoted += gfb;
        }
        if db > 0 || gfb > 0 {
            let down_shape = self.qkv_down_w.shape()?;
            let gf_shape = self.qkv_gain_forget_w.shape()?;
            let dim = self.q_ctrl.dim;
            let slice_stream = |idx: i64| -> Result<LoopController> {
                let down_w = self
                    .qkv_down_w
                    .slice(&[idx, 0, 0], &[idx + 1, down_shape[1], down_shape[2]])?
                    .squeeze(Some(&[0]))?;
                let down_t = down_w.transpose(None)?;
                let gf_w = self
                    .qkv_gain_forget_w
                    .slice(&[idx, 0, 0], &[idx + 1, gf_shape[1], gf_shape[2]])?
                    .squeeze(Some(&[0]))?;
                let gf_t = gf_w.transpose(None)?;
                Ok(LoopController {
                    down_proj_w: down_w,
                    down_proj_t: down_t,
                    gain_forget_w: gf_w,
                    gain_forget_proj_t: gf_t,
                    dim,
                })
            };
            self.q_ctrl = slice_stream(0)?;
            self.k_ctrl = slice_stream(1)?;
            self.v_ctrl = slice_stream(2)?;
        }
        promoted += self.mlp_ctrl.ensure_cuda_device()?;
        Ok(promoted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_attn_res_mix_matches_manual_reference() {
        let d = 4usize;
        let t = 2usize;
        let s0_v = vec![1.0_f32, 2.0, -1.0, 0.5, 0.2, -0.3, 0.4, 0.8];
        let s1_v = vec![-0.5_f32, 0.5, 1.5, -1.0, 1.0, 1.0, -0.5, 0.2];
        let q_raw = [0.3_f32, -0.4, 0.2, 0.5];
        let scale = 1.0_f32 / (d as f32).sqrt();
        let q_scaled: Vec<f32> = q_raw.iter().map(|v| v * scale).collect();

        let s0 = MxArray::from_float32(&s0_v, &[t as i64, d as i64]).unwrap();
        let s1 = MxArray::from_float32(&s1_v, &[t as i64, d as i64]).unwrap();
        let q = MxArray::from_float32(&q_scaled, &[1, d as i64]).unwrap();

        let out = block_attn_res_mix(&[&s0, &s1], &q).unwrap();
        assert_eq!(out.shape().unwrap().as_ref(), &[1, t as i64, d as i64]);
        let got = out.to_float32().unwrap();

        // Manual FP32 reference
        let rms = |row: &[f32]| -> Vec<f32> {
            let mean_sq = row.iter().map(|x| x * x).sum::<f32>() / (row.len() as f32);
            let r = 1.0 / (mean_sq + ATTN_RES_EPS).sqrt();
            row.iter().map(|x| x * r).collect()
        };
        for tok in 0..t {
            let v0 = rms(&s0_v[tok * d..(tok + 1) * d]);
            let v1 = rms(&s1_v[tok * d..(tok + 1) * d]);
            let l0: f32 = v0.iter().zip(&q_scaled).map(|(a, b)| a * b).sum();
            let l1: f32 = v1.iter().zip(&q_scaled).map(|(a, b)| a * b).sum();
            let m = l0.max(l1);
            let e0 = (l0 - m).exp();
            let e1 = (l1 - m).exp();
            let p0 = e0 / (e0 + e1);
            let p1 = e1 / (e0 + e1);
            for j in 0..d {
                let exp_val = p0 * v0[j] + p1 * v1[j];
                assert!(
                    (got[tok * d + j] - exp_val).abs() < 1e-5,
                    "tok={tok} j={j}: got {} expected {exp_val}",
                    got[tok * d + j]
                );
            }
        }
    }

    #[test]
    fn loop_controller_blend_math() {
        let d = 4i64;
        let r = 2i64;
        let dest = MxArray::from_float32(&[1.0, -2.0, 0.5, 1.5], &[1, d]).unwrap();
        let msg = MxArray::from_float32(&[0.5, 1.0, -1.0, 0.25], &[1, d]).unwrap();

        // Zero weights => gain=0, forget=0 => out == dest exactly
        let ctrl_zero = LoopController::new(
            MxArray::zeros(&[r, 2 * d], Some(DType::Float32)).unwrap(),
            MxArray::zeros(&[d, r], Some(DType::Float32)).unwrap(),
            MxArray::zeros(&[d, r], Some(DType::Float32)).unwrap(),
        )
        .unwrap();
        let out = ctrl_zero
            .forward(&dest, &msg)
            .unwrap()
            .to_float32()
            .unwrap();
        let dest_v = dest.to_float32().unwrap();
        for (a, b) in out.iter().zip(dest_v.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
