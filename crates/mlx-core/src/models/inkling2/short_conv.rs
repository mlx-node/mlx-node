use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};

/// Per-visit rolling causal history for Inkling2 ShortConv (`k = 4` -> `k - 1 = 3` rows).
///
/// Stores the last `k - 1` **pre-convolution** rows in oldest-to-newest order:
/// index `0` = `t - (k - 1)` (oldest), index `k - 2` = `t - 1` (newest).
#[derive(Clone)]
pub struct VisitShortConvState {
    pub k: MxArray,
    pub v: MxArray,
    pub attn_post: MxArray,
    pub mlp_post: MxArray,
}

impl VisitShortConvState {
    pub fn zeros(
        history_len: usize,
        kv_dim: usize,
        hidden_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let h = history_len as i64;
        let kv = kv_dim as i64;
        let d = hidden_size as i64;
        Ok(Self {
            k: MxArray::zeros(&[h, kv], Some(dtype))?,
            v: MxArray::zeros(&[h, kv], Some(dtype))?,
            attn_post: MxArray::zeros(&[h, d], Some(dtype))?,
            mlp_post: MxArray::zeros(&[h, d], Some(dtype))?,
        })
    }
}

/// Collection of per-visit ShortConv states across all logical visits (48 for Inkling-2-Mini).
#[derive(Clone)]
pub struct Inkling2ShortConvState {
    pub visits: Vec<VisitShortConvState>,
    pub kernel_size: usize,
}

impl Inkling2ShortConvState {
    pub fn new(
        num_visits: usize,
        kernel_size: usize,
        kv_dim: usize,
        hidden_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        if kernel_size < 2 {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvState: kernel_size must be >= 2, got {kernel_size}"
            )));
        }
        let history_len = kernel_size - 1;
        let mut visits = Vec::with_capacity(num_visits);
        for _ in 0..num_visits {
            visits.push(VisitShortConvState::zeros(
                history_len,
                kv_dim,
                hidden_size,
                dtype,
            )?);
        }
        Ok(Self {
            visits,
            kernel_size,
        })
    }
}

/// Static depthwise causal ShortConv (`k = 4`, residual `+ x[t]`, scale `(k + 1)^(-1/2)`).
///
/// - `x`: `[T, C]` (BF16 or F32)
/// - `weight`: `[C, K]` in **oldest-to-newest** order (`weight[:, 0]` = lag `K-1`, `weight[:, K-1]` = lag `0`).
/// - `history`: mutable `[K - 1, C]` rolling pre-conv state (oldest at row `0`, newest at row `K - 2`).
///
/// Accumulates in FP32 and returns output in `x`'s dtype (BF16).
pub fn static_short_conv(x: &MxArray, weight: &MxArray, history: &mut MxArray) -> Result<MxArray> {
    let x_shape = x.shape()?;
    let w_shape = weight.shape()?;
    if x_shape.len() != 2 || w_shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "static_short_conv: expected 2D x [T, C] and weight [C, K], got {:?} and {:?}",
            x_shape.as_ref(),
            w_shape.as_ref()
        )));
    }
    let t = x_shape[0];
    let c = x_shape[1];
    let k = w_shape[1] as usize;
    if w_shape[0] != c || k < 2 {
        return Err(Error::from_reason(format!(
            "static_short_conv: incompatible weight {:?} for channels {c}",
            w_shape.as_ref()
        )));
    }

    let out_dtype = x.dtype()?;
    let hist_len = (k - 1) as i64;
    let hist_cast = if history.dtype()? != out_dtype {
        history.astype(out_dtype)?
    } else {
        history.clone()
    };

    // `[K - 1 + T, C]`
    let full = MxArray::concatenate(&hist_cast, x, 0)?;
    // Update rolling state to last `K - 1` pre-conv rows.
    *history = full.slice(&[t, 0], &[t + hist_len, c])?;

    let full_f32 = full.astype(DType::Float32)?;
    let w_f32 = if weight.dtype()? == DType::Float32 {
        weight.clone()
    } else {
        weight.astype(DType::Float32)?
    };

    // Residual term: `lag 0` = `x[t]` = `full[K - 1 .. K - 1 + T]`
    let mut acc = full_f32.slice(&[hist_len, 0], &[hist_len + t, c])?;
    for j in 0..k {
        let j_i64 = j as i64;
        // j=0 is oldest tap (lag K-1, starts at 0); j=K-1 is newest tap (lag 0, starts at K-1).
        let x_tap = full_f32.slice(&[j_i64, 0], &[j_i64 + t, c])?;
        let w_tap = w_f32
            .slice(&[0, j_i64], &[c, j_i64 + 1])?
            .reshape(&[1, c])?;
        let term = x_tap.mul(&w_tap)?;
        acc = acc.add(&term)?;
    }

    let scale = 1.0_f64 / ((k as f64) + 1.0).sqrt();
    let scaled = acc.mul_scalar(scale)?;
    if out_dtype != DType::Float32 {
        scaled.astype(out_dtype)
    } else {
        Ok(scaled)
    }
}

/// Dynamic depthwise causal ShortConv for Global layers' K and V streams.
///
/// - `x`: `[T, C]` where `C = num_kv_heads * head_dim`
/// - `static_weight`: `[C, K]` in **oldest-to-newest** order (GGUF layout: `w[:, 0]` = lag `K-1`, `w[:, K-1]` = lag `0`)
/// - `dyn_taps`: `[T, num_kv_heads, K]` in **newest-to-oldest** order (`dyn[:, :, 0]` = lag `0`, `dyn[:, :, K-1]` = lag `K-1`)
/// - `history`: mutable `[K - 1, C]` rolling pre-conv state (oldest at row `0`, newest at row `K - 2`).
///
/// Formula (`scale = (2*K + 1)^(-1/2) = 1/3` for `K = 4`):
/// `y[t, c] = (x[t, c] + sum_{lag=0..K} (w[c, K - 1 - lag] + dyn[t, head, lag]) * x[t - lag, c]) * scale`
pub fn dynamic_short_conv(
    x: &MxArray,
    static_weight: &MxArray,
    dyn_taps: &MxArray,
    history: &mut MxArray,
) -> Result<MxArray> {
    let x_shape = x.shape()?;
    let w_shape = static_weight.shape()?;
    let d_shape = dyn_taps.shape()?;
    if x_shape.len() != 2 || w_shape.len() != 2 || d_shape.len() != 3 {
        return Err(Error::from_reason(format!(
            "dynamic_short_conv: invalid shapes x={:?}, w={:?}, dyn={:?}",
            x_shape.as_ref(),
            w_shape.as_ref(),
            d_shape.as_ref()
        )));
    }
    let t = x_shape[0];
    let c = x_shape[1];
    let k = w_shape[1] as usize;
    let n_kv = d_shape[1];
    if w_shape[0] != c || d_shape[0] != t || d_shape[2] as usize != k || c % n_kv != 0 {
        return Err(Error::from_reason(format!(
            "dynamic_short_conv: incompatible shapes x={:?}, w={:?}, dyn={:?}",
            x_shape.as_ref(),
            w_shape.as_ref(),
            d_shape.as_ref()
        )));
    }
    let head_dim = c / n_kv;
    let out_dtype = x.dtype()?;
    let hist_len = (k - 1) as i64;
    let hist_cast = if history.dtype()? != out_dtype {
        history.astype(out_dtype)?
    } else {
        history.clone()
    };

    let full = MxArray::concatenate(&hist_cast, x, 0)?;
    *history = full.slice(&[t, 0], &[t + hist_len, c])?;

    let full_f32 = full.astype(DType::Float32)?;
    let w_f32 = if static_weight.dtype()? == DType::Float32 {
        static_weight.clone()
    } else {
        static_weight.astype(DType::Float32)?
    };
    let dyn_f32 = if dyn_taps.dtype()? == DType::Float32 {
        dyn_taps.clone()
    } else {
        dyn_taps.astype(DType::Float32)?
    };

    // Residual term: `lag 0` = `x[t]`
    let mut acc = full_f32.slice(&[hist_len, 0], &[hist_len + t, c])?;
    for lag in 0..k {
        let lag_i64 = lag as i64;
        // `full` start offset for `x[t - lag]` is `(K - 1) - lag`.
        let start = hist_len - lag_i64;
        let x_tap = full_f32.slice(&[start, 0], &[start + t, c])?;

        // Static weight is oldest-to-newest, so lag `lag` is column `K - 1 - lag`.
        let w_col = hist_len - lag_i64;
        let w_tap = w_f32
            .slice(&[0, w_col], &[c, w_col + 1])?
            .reshape(&[1, c])?;

        // Dynamic taps are newest-to-oldest (`dyn[:, :, lag]`), broadcast per KV head across `head_dim`.
        let d_tap = dyn_f32
            .slice(&[0, 0, lag_i64], &[t, n_kv, lag_i64 + 1])?
            .reshape(&[t, n_kv, 1])?
            .broadcast_to(&[t, n_kv, head_dim])?
            .reshape(&[t, c])?;

        let eff_w = w_tap.add(&d_tap)?;
        let term = x_tap.mul(&eff_w)?;
        acc = acc.add(&term)?;
    }

    let scale = 1.0_f64 / ((2 * k + 1) as f64).sqrt();
    let scaled = acc.mul_scalar(scale)?;
    if out_dtype != DType::Float32 {
        scaled.astype(out_dtype)
    } else {
        Ok(scaled)
    }
}

/// Fused Metal `k = 4` causal ShortConv (static when `dyn_taps` is `None`, dynamic global when
/// `Some`), updating `history` in place. Falls back to the pure-MLX implementation on non-Metal
/// devices or when `kernel_size != 4`.
pub fn fused_short_conv_k4(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    history: &mut MxArray,
) -> Result<MxArray> {
    #[cfg(target_os = "macos")]
    {
        if unsafe { mlx_sys::mlx_metal_is_available() } && weight.shape()?[1] == 4 {
            let (y, hist_out) = super::kernels::inkling2_sconv_k4(x, weight, history, dyn_taps)?;
            *history = hist_out;
            return Ok(y);
        }
    }
    match dyn_taps {
        Some(dyn_t) => dynamic_short_conv(x, weight, dyn_t, history),
        None => static_short_conv(x, weight, history),
    }
}

/// Per-stream rollback tape for one ShortConv (`k = 4`) stream during speculative verification.
///
/// Captures `hist_before` (`[K-1, C]` BF16) and the `T` pre-convolution candidate rows `rows`
/// (`[T, C]` BF16) written during target verify. On commit of `keep` tokens (`0 <= keep <= T`),
/// [`Self::commit_prefix`] reconstructs the exact `[K-1, C]` rolling history after `keep` tokens
/// with zero corruption from rejected tail tokens!
#[derive(Clone)]
pub struct StreamShortConvTape {
    pub hist_before: MxArray,
    pub rows: MxArray,
}

impl StreamShortConvTape {
    /// Compute the exact `[history_len, C]` rolling ShortConv history after committing `keep`
    /// tokens (`0 <= keep <= rows.shape()[0]`).
    pub fn commit_prefix(&self, keep: usize) -> Result<MxArray> {
        let hist_len = self.hist_before.shape()?[0] as usize;
        let total_rows = self.rows.shape()?[0] as usize;
        let c = self.rows.shape()?[1];
        if keep > total_rows {
            return Err(Error::from_reason(format!(
                "StreamShortConvTape::commit_prefix: keep={keep} exceeds verified rows={total_rows}"
            )));
        }
        if keep == 0 {
            return Ok(self.hist_before.clone());
        }
        if keep >= hist_len {
            let start = (keep - hist_len) as i64;
            return self.rows.slice(&[start, 0], &[keep as i64, c]);
        }
        let hist_tail = self
            .hist_before
            .slice(&[keep as i64, 0], &[hist_len as i64, c])?;
        let rows_head = self.rows.slice(&[0, 0], &[keep as i64, c])?;
        let hist_cast = if hist_tail.dtype()? != rows_head.dtype()? {
            hist_tail.astype(rows_head.dtype()?)?
        } else {
            hist_tail
        };
        MxArray::concatenate(&hist_cast, &rows_head, 0)
    }
}

/// Per-visit ShortConv rollback tape across all 4 streams (`k`, `v`, `attn_post`, `mlp_post`).
#[derive(Clone)]
pub struct VisitShortConvTape {
    pub k: Option<StreamShortConvTape>,
    pub v: Option<StreamShortConvTape>,
    pub attn_post: Option<StreamShortConvTape>,
    pub mlp_post: Option<StreamShortConvTape>,
}

impl VisitShortConvTape {
    pub fn empty() -> Self {
        Self {
            k: None,
            v: None,
            attn_post: None,
            mlp_post: None,
        }
    }

    pub fn commit_prefix(&self, keep: usize, dst: &mut VisitShortConvState) -> Result<()> {
        if let Some(ref t) = self.k {
            dst.k = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.v {
            dst.v = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.attn_post {
            dst.attn_post = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.mlp_post {
            dst.mlp_post = t.commit_prefix(keep)?;
        }
        Ok(())
    }
}

/// Complete 48-visit ShortConv rollback tape for one speculative verify cycle.
#[derive(Clone)]
pub struct Inkling2ShortConvTape {
    pub visits: Vec<VisitShortConvTape>,
}

impl Inkling2ShortConvTape {
    pub fn new(num_visits: usize) -> Self {
        Self {
            visits: (0..num_visits)
                .map(|_| VisitShortConvTape::empty())
                .collect(),
        }
    }

    /// Commit ONLY the accepted prefix (`0..=verify_rows`) into `state` across all 48 visits.
    pub fn commit_prefix(&self, keep: usize, state: &mut Inkling2ShortConvState) -> Result<()> {
        if self.visits.len() != state.visits.len() {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvTape visit count mismatch: tape={} state={}",
                self.visits.len(),
                state.visits.len()
            )));
        }
        for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
            tape_v.commit_prefix(keep, state_v)?;
        }
        Ok(())
    }
}

/// Apply fused ShortConv over a packed multi-sequence batch partitioned by `cu_seqlens_q`
/// (`[0, len_0, len_0 + len_1, ...]`), updating each sequence's history independently.
///
/// For the common single-sequence case (`histories.len() == 1`), calls [`fused_short_conv_k4`]
/// directly with zero slice/concat overhead.
pub fn fused_short_conv_ragged(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
) -> Result<MxArray> {
    fused_short_conv_ragged_taped(x, weight, dyn_taps, histories, cu_seqlens_q, None)
}

/// Same as [`fused_short_conv_ragged`], additionally recording each sequence's pre-conv rows and
/// pre-verify history into `tapes` (one `Option<&mut Option<StreamShortConvTape>>` per sequence)
/// when speculative rollback is requested.
pub fn fused_short_conv_ragged_taped(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
) -> Result<MxArray> {
    fused_short_conv_ragged_residual_taped(
        x,
        weight,
        dyn_taps,
        None,
        None,
        histories,
        cu_seqlens_q,
        None,
        tapes,
    )
}

/// Single-dispatch ragged causal `k=4` ShortConv across `N >= 1` packed sequences with optional
/// FP32 residual epilogue (`res_a` + optional `res_b` + `conv_out`) and optional speculative
/// rollback tape capture (`hist_before`, pre-conv `rows`).
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_ragged_residual_taped(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    res_a: Option<&MxArray>,
    res_b: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
    mut tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
) -> Result<MxArray> {
    let num_seqs = histories.len();
    if num_seqs == 0 || cu_seqlens_q.len() != num_seqs + 1 {
        return Err(Error::from_reason(format!(
            "fused_short_conv_ragged: invalid seq count {num_seqs} vs cu_seqlens_q len {}",
            cu_seqlens_q.len()
        )));
    }
    let c = x.shape()?[1];
    let all_non_empty = (0..num_seqs).all(|s| cu_seqlens_q[s + 1] > cu_seqlens_q[s]);

    #[cfg(target_os = "macos")]
    if unsafe { mlx_sys::mlx_metal_is_available() } && weight.shape()?[1] == 4 && all_non_empty {
        if let Some(ref mut tape_slots) = tapes {
            if num_seqs == 1 {
                *tape_slots[0] = Some(StreamShortConvTape {
                    hist_before: histories[0].clone(),
                    rows: x.clone(),
                });
            } else {
                for (seq_idx, hist) in histories.iter().enumerate() {
                    let start = cu_seqlens_q[seq_idx] as i64;
                    let end = cu_seqlens_q[seq_idx + 1] as i64;
                    let x_seq = x.slice(&[start, 0], &[end, c])?;
                    *tape_slots[seq_idx] = Some(StreamShortConvTape {
                        hist_before: (*hist).clone(),
                        rows: x_seq,
                    });
                }
            }
        }
        if num_seqs == 1 && res_a.is_none() && res_b.is_none() {
            let (y, hist_out) =
                super::kernels::inkling2_sconv_k4(x, weight, histories[0], dyn_taps)?;
            *histories[0] = hist_out;
            return Ok(y);
        }
        let cu_owned;
        let cu_ref = if num_seqs > 1 {
            match cu_seqlens_arr {
                Some(arr) => Some(arr),
                None => {
                    let vals: Vec<i32> = cu_seqlens_q.iter().map(|&v| v as i32).collect();
                    cu_owned = MxArray::from_int32(&vals, &[vals.len() as i64])?;
                    Some(&cu_owned)
                }
            }
        } else {
            None
        };
        let (y, hist_out) = if num_seqs <= 4 {
            let h_refs: Vec<&MxArray> = histories.iter().map(|h| &**h).collect();
            super::kernels::inkling2_sconv_k4_ragged(
                x, weight, dyn_taps, cu_ref, res_a, res_b, &h_refs, num_seqs, false,
            )?
        } else {
            let h_refs: Vec<&MxArray> = histories.iter().map(|h| &**h).collect();
            let packed = MxArray::stack(h_refs, Some(0))?;
            super::kernels::inkling2_sconv_k4_ragged(
                x,
                weight,
                dyn_taps,
                cu_ref,
                res_a,
                res_b,
                &[&packed],
                num_seqs,
                true,
            )?
        };
        if num_seqs == 1 {
            *histories[0] = hist_out;
        } else {
            for (seq_idx, hist) in histories.iter_mut().enumerate() {
                let s_i64 = seq_idx as i64;
                **hist = hist_out
                    .slice(&[s_i64, 0, 0], &[s_i64 + 1, 3, c])?
                    .squeeze(Some(&[0]))?;
            }
        }
        return Ok(y);
    }

    // Fallback (non-Metal, kernel_size != 4, or empty sequences)
    let act_dtype = x.dtype()?;
    let y = if num_seqs == 1 {
        if let Some(ref mut tape_slots) = tapes {
            *tape_slots[0] = Some(StreamShortConvTape {
                hist_before: histories[0].clone(),
                rows: x.clone(),
            });
        }
        fused_short_conv_k4(x, weight, dyn_taps, histories[0])?
    } else {
        let dyn_dims = match dyn_taps {
            Some(d) => {
                let s = d.shape()?;
                Some((s[1], s[2]))
            }
            None => None,
        };
        let mut outputs = Vec::with_capacity(num_seqs);
        for (seq_idx, hist) in histories.iter_mut().enumerate() {
            let start = cu_seqlens_q[seq_idx] as i64;
            let end = cu_seqlens_q[seq_idx + 1] as i64;
            if end <= start {
                continue;
            }
            let x_seq = x.slice(&[start, 0], &[end, c])?;
            if let Some(ref mut tape_slots) = tapes {
                *tape_slots[seq_idx] = Some(StreamShortConvTape {
                    hist_before: hist.clone(),
                    rows: x_seq.clone(),
                });
            }
            let d_seq = match (dyn_taps, dyn_dims) {
                (Some(d), Some((n_kv, k))) => Some(d.slice(&[start, 0, 0], &[end, n_kv, k])?),
                _ => None,
            };
            let y_seq = fused_short_conv_k4(&x_seq, weight, d_seq.as_ref(), hist)?;
            outputs.push(y_seq);
        }
        let refs: Vec<&MxArray> = outputs.iter().collect();
        MxArray::concatenate_many(refs, Some(0))?
    };
    match (res_a, res_b) {
        (None, None) => Ok(y),
        (Some(a), None) => a
            .astype(DType::Float32)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
        (None, Some(b)) => b
            .astype(DType::Float32)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
        (Some(a), Some(b)) => a
            .astype(DType::Float32)?
            .add(&b.astype(DType::Float32)?)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
    }
}

/// Single-dispatch fused K+V causal `k=4` ShortConv + K L2-clip + V weightless RMSNorm across
/// `N >= 1` packed sequences with zero DRAM roundtrip between ShortConv and K/V norm/clip.
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_kv_prep_taped(
    k_raw: &MxArray,
    v_raw: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
    mut k_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    mut v_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    num_kv_heads: usize,
    head_dim: usize,
    qk_thresh: f32,
    v_eps: f32,
) -> Result<(MxArray, MxArray)> {
    let num_seqs = k_histories.len();
    if num_seqs == 0 || v_histories.len() != num_seqs || cu_seqlens_q.len() != num_seqs + 1 {
        return Err(Error::from_reason(
            "fused_short_conv_kv_prep_taped: invalid sequence count",
        ));
    }
    let kv_dim = (num_kv_heads * head_dim) as i64;
    let all_non_empty = (0..num_seqs).all(|s| cu_seqlens_q[s + 1] > cu_seqlens_q[s]);

    #[cfg(target_os = "macos")]
    if unsafe { mlx_sys::mlx_metal_is_available() }
        && k_weight.shape()?[1] == 4
        && v_weight.shape()?[1] == 4
        && all_non_empty
    {
        if let Some(ref mut kt) = k_tapes {
            if num_seqs == 1 {
                *kt[0] = Some(StreamShortConvTape {
                    hist_before: k_histories[0].clone(),
                    rows: k_raw.clone(),
                });
            } else {
                for (seq_idx, hist) in k_histories.iter().enumerate() {
                    let st = cu_seqlens_q[seq_idx] as i64;
                    let en = cu_seqlens_q[seq_idx + 1] as i64;
                    *kt[seq_idx] = Some(StreamShortConvTape {
                        hist_before: (*hist).clone(),
                        rows: k_raw.slice(&[st, 0], &[en, kv_dim])?,
                    });
                }
            }
        }
        if let Some(ref mut vt) = v_tapes {
            if num_seqs == 1 {
                *vt[0] = Some(StreamShortConvTape {
                    hist_before: v_histories[0].clone(),
                    rows: v_raw.clone(),
                });
            } else {
                for (seq_idx, hist) in v_histories.iter().enumerate() {
                    let st = cu_seqlens_q[seq_idx] as i64;
                    let en = cu_seqlens_q[seq_idx + 1] as i64;
                    *vt[seq_idx] = Some(StreamShortConvTape {
                        hist_before: (*hist).clone(),
                        rows: v_raw.slice(&[st, 0], &[en, kv_dim])?,
                    });
                }
            }
        }
        let cu_owned;
        let cu_ref = if num_seqs > 1 {
            match cu_seqlens_arr {
                Some(arr) => Some(arr),
                None => {
                    let vals: Vec<i32> = cu_seqlens_q.iter().map(|&v| v as i32).collect();
                    cu_owned = MxArray::from_int32(&vals, &[vals.len() as i64])?;
                    Some(&cu_owned)
                }
            }
        } else {
            None
        };
        let (k_cur, v_cur, kh_out, vh_out) = if num_seqs <= 4 {
            let kh_refs: Vec<&MxArray> = k_histories.iter().map(|h| &**h).collect();
            let vh_refs: Vec<&MxArray> = v_histories.iter().map(|h| &**h).collect();
            super::kernels::inkling2_sconv_k4_kv_prep(
                k_raw,
                v_raw,
                k_weight,
                v_weight,
                dyn_k,
                dyn_v,
                cu_ref,
                &kh_refs,
                &vh_refs,
                num_seqs,
                false,
                num_kv_heads,
                head_dim,
                qk_thresh,
                v_eps,
                1e-6,
            )?
        } else {
            let kh_refs: Vec<&MxArray> = k_histories.iter().map(|h| &**h).collect();
            let vh_refs: Vec<&MxArray> = v_histories.iter().map(|h| &**h).collect();
            let kh_p = MxArray::stack(kh_refs, Some(0))?;
            let vh_p = MxArray::stack(vh_refs, Some(0))?;
            super::kernels::inkling2_sconv_k4_kv_prep(
                k_raw,
                v_raw,
                k_weight,
                v_weight,
                dyn_k,
                dyn_v,
                cu_ref,
                &[&kh_p],
                &[&vh_p],
                num_seqs,
                true,
                num_kv_heads,
                head_dim,
                qk_thresh,
                v_eps,
                1e-6,
            )?
        };
        if num_seqs == 1 {
            *k_histories[0] = kh_out;
            *v_histories[0] = vh_out;
        } else {
            for (seq_idx, (kh, vh)) in k_histories
                .iter_mut()
                .zip(v_histories.iter_mut())
                .enumerate()
            {
                let s_i64 = seq_idx as i64;
                **kh = kh_out
                    .slice(&[s_i64, 0, 0], &[s_i64 + 1, 3, kv_dim])?
                    .squeeze(Some(&[0]))?;
                **vh = vh_out
                    .slice(&[s_i64, 0, 0], &[s_i64 + 1, 3, kv_dim])?
                    .squeeze(Some(&[0]))?;
            }
        }
        return Ok((k_cur, v_cur));
    }

    // Fallback
    let t_i64 = k_raw.shape()?[0];
    let act_dtype = k_raw.dtype()?;
    let k_conv_out =
        fused_short_conv_ragged_taped(k_raw, k_weight, dyn_k, k_histories, cu_seqlens_q, k_tapes)?;
    let v_conv_out =
        fused_short_conv_ragged_taped(v_raw, v_weight, dyn_v, v_histories, cu_seqlens_q, v_tapes)?;
    let k_3d = k_conv_out.reshape(&[t_i64, num_kv_heads as i64, head_dim as i64])?;
    let v_3d = v_conv_out.reshape(&[t_i64, num_kv_heads as i64, head_dim as i64])?;
    let k_cur = super::attention::l2_clip(&k_3d, None, f64::from(qk_thresh), act_dtype)?;
    let v_cur = crate::nn::normalization::rms_norm_unscaled(&v_3d.astype(DType::Float32)?, v_eps)?
        .astype(act_dtype)?;
    Ok((k_cur, v_cur))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_short_conv_prefill_matches_step_by_step_decode() {
        let t = 5;
        let c = 4;
        let k = 4;
        let x_vals: Vec<f32> = (0..(t * c)).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Oldest-to-newest weights
        let w_vals: Vec<f32> = (0..(c * k)).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();

        let x = MxArray::from_float32(&x_vals, &[t as i64, c as i64]).unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();

        // Prefill all T tokens at once
        let mut hist_pre =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let out_pre = static_short_conv(&x, &w, &mut hist_pre).unwrap();
        let pre_vec = out_pre.to_float32().unwrap();

        // Step-by-step decode 1 token at a time
        let mut hist_dec =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let mut dec_vec = Vec::new();
        for step in 0..t {
            let x_step = x
                .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                .unwrap();
            let y_step = static_short_conv(&x_step, &w, &mut hist_dec).unwrap();
            dec_vec.extend_from_slice(&y_step.to_float32().unwrap());
        }

        for (a, b) in pre_vec.iter().zip(dec_vec.iter()) {
            assert!((a - b).abs() < 1e-5, "prefill {a} != decode {b}");
        }

        // Verify exact closed-form value at t=0 (all history is zero):
        // y[0, ch] = (x[0, ch] + w[ch, 3] * x[0, ch]) / sqrt(5)
        let inv_sqrt5 = 1.0_f32 / 5.0_f32.sqrt();
        for ch in 0..c {
            let expected = (x_vals[ch] + w_vals[ch * k + 3] * x_vals[ch]) * inv_sqrt5;
            assert!(
                (pre_vec[ch] - expected).abs() < 1e-5,
                "ch={ch}: got {} expected {expected}",
                pre_vec[ch]
            );
        }
    }

    #[test]
    fn dynamic_short_conv_tap_order_and_prefill_decode_parity() {
        let t = 5;
        let n_kv = 2;
        let head_dim = 2;
        let c = n_kv * head_dim;
        let k = 4;

        let x_vals: Vec<f32> = (0..(t * c)).map(|i| (i as f32 + 1.0) * 0.15).collect();
        let w_vals: Vec<f32> = (0..(c * k))
            .map(|i| ((i % 5) as f32 - 2.0) * 0.25)
            .collect();
        let dyn_vals: Vec<f32> = (0..(t * n_kv * k))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();

        let x = MxArray::from_float32(&x_vals, &[t as i64, c as i64]).unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();
        let dyn_taps =
            MxArray::from_float32(&dyn_vals, &[t as i64, n_kv as i64, k as i64]).unwrap();

        let mut hist_pre =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let out_pre = dynamic_short_conv(&x, &w, &dyn_taps, &mut hist_pre).unwrap();
        let pre_vec = out_pre.to_float32().unwrap();

        let mut hist_dec =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let mut dec_vec = Vec::new();
        for step in 0..t {
            let x_step = x
                .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                .unwrap();
            let d_step = dyn_taps
                .slice(
                    &[step as i64, 0, 0],
                    &[(step + 1) as i64, n_kv as i64, k as i64],
                )
                .unwrap();
            let y_step = dynamic_short_conv(&x_step, &w, &d_step, &mut hist_dec).unwrap();
            dec_vec.extend_from_slice(&y_step.to_float32().unwrap());
        }

        for (a, b) in pre_vec.iter().zip(dec_vec.iter()) {
            assert!((a - b).abs() < 1e-5, "prefill {a} != decode {b}");
        }

        // Verify closed-form at t=3 (full 4 lags present):
        // lag l uses w[ch, 3 - l] + dyn[3, ch / head_dim, l]
        for ch in 0..c {
            let h = ch / head_dim;
            let mut sum = x_vals[3 * c + ch];
            for lag in 0..k {
                let w_l = w_vals[ch * k + (k - 1 - lag)];
                let d_l = dyn_vals[(3 * n_kv + h) * k + lag];
                let x_l = x_vals[(3 - lag) * c + ch];
                sum += (w_l + d_l) * x_l;
            }
            let expected = sum / 3.0;
            assert!(
                (pre_vec[3 * c + ch] - expected).abs() < 1e-5,
                "t=3 ch={ch}: got {} expected {expected}",
                pre_vec[3 * c + ch]
            );
        }
    }

    #[test]
    fn short_conv_tape_commit_prefix_exact_for_all_keep_lengths() {
        let c = 8usize;
        let k = 4usize;
        let hist_len = k - 1;
        let verify_rows = 8usize;

        let init_hist_vals: Vec<f32> = (0..(hist_len * c))
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        let x_vals: Vec<f32> = (0..(verify_rows * c))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let w_vals: Vec<f32> = (0..(c * k)).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

        let init_hist = MxArray::from_float32(&init_hist_vals, &[hist_len as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let x = MxArray::from_float32(&x_vals, &[verify_rows as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();

        let mut verify_hist = init_hist.clone();
        let mut tape_slot: Option<StreamShortConvTape> = None;
        let _y = fused_short_conv_ragged_taped(
            &x,
            &w,
            None,
            &mut [&mut verify_hist],
            &[0, verify_rows],
            Some(&mut [&mut tape_slot]),
        )
        .unwrap();
        let tape = tape_slot.expect("tape recorded");

        // Compare `tape.commit_prefix(keep)` against step-by-step exact history for EVERY keep in 0..=8
        for keep in 0..=verify_rows {
            let mut ref_hist = init_hist.clone();
            for step in 0..keep {
                let x_row = x
                    .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                    .unwrap();
                let _ = static_short_conv(&x_row, &w, &mut ref_hist).unwrap();
            }
            let got = tape
                .commit_prefix(keep)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let exp = ref_hist
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(got.len(), exp.len());
            for (idx, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "keep={keep} idx={idx}: tape {a} != ref {b}"
                );
            }
        }
    }
}
