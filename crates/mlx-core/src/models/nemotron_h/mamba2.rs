//! NemotronH Mamba-2 SSM mixer: in_proj (gate | xBC | dt), depthwise causal
//! conv1d, the SSM itself (chunk scan on prefill, recurrent update on decode),
//! a Zamba2 gated RMSNorm, then out_proj.
//!
//! The conv state, dt, A and the scan are kept in f32 while the references
//! carry them in the model dtype, so `gated_rmsnorm` has to restore the mixer's
//! entry dtype at the exit boundary.

use crate::array::{DType, MxArray};
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Activations, Conv1d};
use napi::bindgen_prelude::*;

use super::config::NemotronHConfig;

/// Persistent Mamba-2 state for one request.
#[derive(Clone)]
pub struct Mamba2State {
    /// Conv state [B, conv_kernel - 1, conv_dim].
    pub conv: MxArray,
    /// SSM recurrent state [B, num_heads, head_dim, state_size] f32.
    pub ssm: MxArray,
}

impl Mamba2State {
    /// Stack per-request rows along the batch axis. All rows must share the
    /// same non-batch shape.
    pub(crate) fn stack_rows(rows: &[&Mamba2State]) -> Result<Mamba2State> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "Mamba2State::stack_rows requires at least one row",
            ));
        }
        if rows.len() == 1 {
            return Ok(rows[0].clone());
        }
        let convs = rows.iter().map(|s| &s.conv).collect::<Vec<_>>();
        let ssms = rows.iter().map(|s| &s.ssm).collect::<Vec<_>>();
        Ok(Mamba2State {
            conv: MxArray::concatenate_many(convs, Some(0))?,
            ssm: MxArray::concatenate_many(ssms, Some(0))?,
        })
    }

    /// Extract row `index` (batch axis) as a fresh single-request state.
    pub(crate) fn row(&self, index: i64) -> Result<Mamba2State> {
        Ok(Mamba2State {
            conv: self.conv.slice_axis(0, index, index + 1)?,
            ssm: self.ssm.slice_axis(0, index, index + 1)?,
        })
    }
}

/// The Mamba-2 mixer. A_log/D/dt_bias and the gated-norm weight are kept as
/// plain tensors, never quantized.
pub struct NemotronHMamba2Mixer {
    pub(crate) in_proj: LinearProj,
    pub(crate) conv1d: Conv1d,
    pub(crate) dt_bias: MxArray,
    pub(crate) a_log: MxArray,
    pub(crate) d: MxArray,
    pub(crate) norm_weight: MxArray,
    pub(crate) out_proj: LinearProj,
    pub(crate) num_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) d_inner: i32,
    pub(crate) n_groups: i32,
    pub(crate) ssm_state_size: i32,
    pub(crate) conv_kernel: i32,
    pub(crate) chunk_size: i32,
    /// `(min, max)` bounds applied to `softplus(dt + dt_bias)`, from
    /// `NemotronHConfig::time_step_limit_pair()`. Defaults to the unbounded
    /// `(0.0, +inf)` as every served reference does, and is NOT derived from
    /// `time_step_min`.
    pub(crate) dt_limit: (f64, f64),
    /// Epsilon for the Zamba2 gated RMSNorm, from
    /// `NemotronHConfig::layer_norm_epsilon` — never a hardcoded literal, or a
    /// sibling checkpoint declaring a different one is normalized wrongly.
    pub(crate) norm_eps: f64,
}

/// Build a fresh (random-initialized) Mamba mixer sized from config.
pub fn new_mamba_mixer(config: &NemotronHConfig) -> Result<NemotronHMamba2Mixer> {
    let h = config.hidden_size as u32;
    let d_inner = config.mamba_intermediate_size() as u32;
    let conv_dim = config.mamba_conv_dim() as u32;
    let num_heads = config.mamba_num_heads;

    let in_proj = LinearProj::Standard(crate::nn::Linear::new(
        h,
        d_inner + conv_dim + num_heads as u32,
        Some(false),
    )?);
    let conv1d = Conv1d::new(
        conv_dim,
        conv_dim,
        config.conv_kernel as u32,
        None,
        None,
        None,
        Some(conv_dim),
        Some(true),
    )?;
    let out_proj = LinearProj::Standard(crate::nn::Linear::new(d_inner, h, Some(false))?);

    Ok(NemotronHMamba2Mixer {
        in_proj,
        conv1d,
        dt_bias: MxArray::zeros(&[num_heads as i64], Some(DType::Float32))?,
        a_log: MxArray::zeros(&[num_heads as i64], Some(DType::Float32))?,
        d: MxArray::zeros(&[num_heads as i64], Some(DType::Float32))?,
        norm_weight: MxArray::ones(&[d_inner as i64], Some(DType::Float32))?,
        out_proj,
        num_heads,
        head_dim: config.mamba_head_dim,
        d_inner: d_inner as i32,
        n_groups: config.n_groups,
        ssm_state_size: config.ssm_state_size,
        conv_kernel: config.conv_kernel,
        chunk_size: config.chunk_size,
        dt_limit: config.time_step_limit_pair(),
        norm_eps: config.layer_norm_epsilon,
    })
}

impl NemotronHMamba2Mixer {
    /// The Mamba conv dim: d_inner + 2 * n_groups * ssm_state_size.
    fn conv_dim(&self) -> i32 {
        self.d_inner + 2 * self.n_groups * self.ssm_state_size
    }

    pub fn is_quantized(&self) -> bool {
        self.in_proj.is_quantized() || self.out_proj.is_quantized()
    }

    pub fn fresh_state(&self, batch: i64) -> Result<Mamba2State> {
        let conv_dim = self.conv1d.get_weight().shape_at(0)?;
        Ok(Mamba2State {
            conv: MxArray::zeros(
                &[batch, (self.conv_kernel - 1).max(0) as i64, conv_dim],
                Some(DType::Float32),
            )?,
            ssm: MxArray::zeros(
                &[
                    batch,
                    self.num_heads as i64,
                    self.head_dim as i64,
                    self.ssm_state_size as i64,
                ],
                Some(DType::Float32),
            )?,
        })
    }

    /// State present + T == 1 takes the recurrent decode path; otherwise the
    /// chunk scan, seeded from any present state for a continuation prefill.
    pub fn forward(&self, x: &MxArray, state: Option<&mut Mamba2State>) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq = shape[1];
        // The mixer is dtype-transparent in every reference, but this port
        // keeps the conv state, dt, A and the scan in f32 — so capture the
        // entry dtype HERE for `gated_rmsnorm` to re-apply on the way out.
        let io_dtype = x.dtype()?;
        let projected = self.in_proj.forward(x)?;
        // Split by explicit sizes — the three sections are unequal:
        // gate [d_inner] | xBC [conv_dim] | dt [num_heads].
        let d_inner = self.d_inner as i64;
        let conv_dim = (self.conv_dim()) as i64;
        let gate = projected.slice_axis(2, 0, d_inner)?;
        let xbc = projected.slice_axis(2, d_inner, d_inner + conv_dim)?;
        let dt_raw = projected.slice_axis(
            2,
            d_inner + conv_dim,
            d_inner + conv_dim + self.num_heads as i64,
        )?;

        let is_decode = state.is_some() && seq == 1;
        let (conv_out, next_conv) =
            self.conv_forward(&xbc, state.as_ref().map(|s| &s.conv), is_decode)?;

        let d_inner = self.d_inner as i64;
        let group_state = (self.n_groups * self.ssm_state_size) as i64;
        let x_inner = conv_out.slice_axis(2, 0, d_inner)?;
        let b_groups = conv_out
            .slice_axis(2, d_inner, d_inner + group_state)?
            .reshape(&[batch, seq, self.n_groups as i64, self.ssm_state_size as i64])?;
        let c_groups = conv_out
            .slice_axis(2, d_inner + group_state, d_inner + 2 * group_state)?
            .reshape(&[batch, seq, self.n_groups as i64, self.ssm_state_size as i64])?;

        // dt = clip(softplus(dt + dt_bias), time_step_limit), which defaults to
        // no clamp. `time_step_min` is deliberately not consulted.
        let dt = Activations::softplus(
            &dt_raw
                .astype(DType::Float32)?
                .add(&self.dt_bias.astype(DType::Float32)?)?,
        )?
        .clip(Some(self.dt_limit.0), Some(self.dt_limit.1))?;

        let a = self.a_log.astype(DType::Float32)?.exp()?.mul_scalar(-1.0)?;

        let x_reshaped =
            x_inner.reshape(&[batch, seq, self.num_heads as i64, self.head_dim as i64])?;

        let (scan_out, next_ssm) = if is_decode {
            let prev = state
                .as_ref()
                .ok_or_else(|| Error::from_reason("decode path requires a state"))?
                .ssm
                .clone();
            self.decode_step(&x_reshaped, &dt, &a, &b_groups, &c_groups, &prev)?
        } else {
            let init = state.as_ref().map(|s| s.ssm.clone());
            chunk_scan(
                &x_reshaped,
                &dt,
                &a,
                &b_groups,
                &c_groups,
                &self.d,
                self.chunk_size as i64,
                init.as_ref(),
            )?
        };

        let scan_flat = scan_out.reshape(&[batch, seq, self.d_inner as i64])?;
        let normed = gated_rmsnorm(
            &scan_flat,
            &gate,
            &self.norm_weight,
            self.norm_eps,
            (self.d_inner / self.n_groups) as i64,
            io_dtype,
        )?;
        let out = self.out_proj.forward(&normed)?;

        if let Some(st) = state {
            st.conv = next_conv;
            st.ssm = next_ssm;
        }
        Ok(out)
    }
}

impl NemotronHMamba2Mixer {
    /// Depthwise causal conv1d over xbc [B, T, conv_dim] with explicit
    /// state [B, conv_kernel-1, conv_dim]. Returns (output, next_state).
    fn conv_forward(
        &self,
        xbc: &MxArray,
        state: Option<&MxArray>,
        is_decode: bool,
    ) -> Result<(MxArray, MxArray)> {
        let padded = if let Some(state) = state {
            MxArray::concatenate(state, xbc, 1)?
        } else {
            let pad = self.conv_kernel - 1;
            xbc.pad(&[0, 0, pad, 0, 0, 0], 0.0)?
        };
        let total_len = padded.shape_at(1)?;
        let keep = (self.conv_kernel - 1) as i64;
        let next_state = padded.slice_axis(1, total_len - keep, total_len)?;
        let conv_out = self.conv1d.forward(&padded)?;
        // The mixer activation (SiLU) belongs on the conv output BEFORE the SSM
        // split into x/B/C. Without it, negative pre-activation values reach the
        // SSM, the recurrent state accumulates unbounded and output diverges.
        let conv_out = Activations::silu(&conv_out)?;
        let out = if is_decode {
            let t = conv_out.shape_at(1)?;
            conv_out.slice_axis(1, t - 1, t)?
        } else {
            conv_out
        };
        Ok((out, next_state))
    }

    /// Recurrent decode: one token, per-head update. All inputs are f32, with
    /// `dt` already softplus'd and clamped and B/C still in group form.
    fn decode_step(
        &self,
        x: &MxArray,
        dt: &MxArray,
        a: &MxArray,
        b: &MxArray,
        c: &MxArray,
        state: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let hg = self.num_heads / self.n_groups;
        let x_bhd = x.squeeze(Some(&[1]))?;
        let dt_bhd = dt.squeeze(Some(&[1]))?;
        let b_head = expand_groups(b, hg as i64)?;
        let c_head = expand_groups(c, hg as i64)?;

        // dA = exp(dt * A)  [B,H,D,NS]
        let dt4 = dt_bhd.reshape(&[batch, self.num_heads as i64, 1, 1])?;
        let a4 = a.reshape(&[1, self.num_heads as i64, 1, 1])?;
        let d_a = dt4.mul(&a4)?.exp()?;
        let d_a = d_a.broadcast_to(&[
            batch,
            self.num_heads as i64,
            self.head_dim as i64,
            self.ssm_state_size as i64,
        ])?;

        let dt3 = dt4.broadcast_to(&[batch, self.num_heads as i64, self.head_dim as i64, 1])?;
        let b4 = b_head.reshape(&[batch, self.num_heads as i64, 1, self.ssm_state_size as i64])?;
        let db = dt3.mul(&b4)?;
        let x4 = x_bhd.reshape(&[batch, self.num_heads as i64, self.head_dim as i64, 1])?;
        let dbx = db.mul(&x4)?;

        let next_state = state.mul(&d_a)?.add(&dbx)?;

        // out = sum_NS next_state * C[..., None] + D * x
        let c4 = c_head.reshape(&[batch, self.num_heads as i64, 1, self.ssm_state_size as i64])?;
        let out = next_state.mul(&c4)?.sum(Some(&[-1]), None)?;
        let d_res = self
            .d
            .reshape(&[1, self.num_heads as i64, 1])?
            .mul(&x_bhd)?;
        let out = out.add(&d_res)?;

        let out = out.reshape(&[batch, 1, (self.num_heads * self.head_dim) as i64])?;
        Ok((out, next_state))
    }
}

/// Expand group-dimension tensors [..., G, NS] to per-head [..., G*HG, NS]
/// by repeating each group HG times (torch repeat_interleave on dim -2).
fn expand_groups(v: &MxArray, heads_per_group: i64) -> Result<MxArray> {
    let shape = v.shape()?;
    let nd = shape.len();
    let ns = shape[nd - 1];
    let g = shape[nd - 2];
    let mut mid: Vec<i64> = shape[..nd - 2].to_vec();
    mid.push(g);
    mid.push(heads_per_group);
    mid.push(ns);
    let expanded = v.expand_dims((nd - 1) as i32)?.broadcast_to(&mid)?;
    let mut out_shape: Vec<i64> = shape[..nd - 2].to_vec();
    out_shape.push(g * heads_per_group);
    out_shape.push(ns);
    expanded.reshape(&out_shape)
}

/// Lower-triangular boolean masks for a chunk x chunk matrix:
/// (strict: j < i, diag: j <= i).
fn triangular_masks(chunk: i64) -> Result<(MxArray, MxArray)> {
    let rows: Vec<i32> = (0..chunk).map(|i| i as i32).collect();
    let r = MxArray::from_int32(&rows, &[chunk, 1])?.astype(DType::Float32)?;
    let c = MxArray::from_int32(&rows, &[1, chunk])?.astype(DType::Float32)?;
    let strict = r.greater(&c)?;
    let diag = r.greater_equal(&c)?;
    Ok((strict, diag))
}

/// Stable segment sum over the last dim (torch segment_sum): for
/// x [..., C] returns [..., C, C] with out[i, j] = sum_{k=j+1..i} x[k]
/// for j <= i and -inf above the diagonal.
fn segment_sum(x: &MxArray, chunk: i64) -> Result<MxArray> {
    let (strict, diag) = triangular_masks(chunk)?;
    let mut shape = x.shape()?.to_vec();
    shape.push(chunk);
    let expanded = x.expand_dims(-1)?.broadcast_to(&shape)?;
    let zero = MxArray::scalar_float(0.0)?.astype(x.dtype()?)?;
    let masked = strict.where_(&expanded, &zero)?;
    let segsum = masked.cumsum(-2)?;
    let ninf = MxArray::scalar_float(-1.0e30)?.astype(x.dtype()?)?;
    diag.where_(&segsum, &ninf)
}

/// Zamba2 gated RMSNorm: (x * silu(gate)) reshaped [..., groups, gs],
/// normalized per group (mean of squares + eps), rescaled by weight.
///
/// `out_dtype` is the MIXER's I/O dtype, NOT `x`'s — `x` is the f32 SSM scan
/// output. This is where the f32 accumulation is closed; let it escape and
/// `out_proj` inherits it, the residual stream promotes at the first Mamba
/// layer, and the flat KV cache ends up f32 against a 2-byte paged pool.
///
/// `weight` is cast too, since the loader upcasts the checkpoint's BF16
/// `mixer.norm.weight` and an f32 weight would re-promote the product
/// (bf16 -> f32 -> bf16 is lossless, so this recovers the checkpoint bytes).
fn gated_rmsnorm(
    x: &MxArray,
    gate: &MxArray,
    weight: &MxArray,
    eps: f64,
    group_size: i64,
    out_dtype: DType,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let nd = shape.len();
    let last = shape[nd - 1];
    let groups = last / group_size;
    let gate_silu = Activations::silu(&gate.astype(DType::Float32)?)?;
    let h = x.astype(DType::Float32)?.mul(&gate_silu)?;
    let mut g_shape = shape.to_vec();
    g_shape.pop();
    g_shape.push(groups);
    g_shape.push(group_size);
    let hg = h.reshape(&g_shape)?;
    let var = hg.square()?.mean(Some(&[-1]), Some(true))?;
    let denom = MxArray::scalar_float(eps)?.add(&var)?;
    let rstd = MxArray::scalar_float(1.0)?.div(&denom.sqrt()?)?;
    let normed = hg.mul(&rstd)?.reshape(&shape)?.astype(out_dtype)?;
    weight.astype(out_dtype)?.mul(&normed)
}

/// Left-pad a tensor along the time axis (axis 1) with pad_size zeros.
fn pad_time(v: &MxArray, pad_size: i64, ndim: i64) -> Result<MxArray> {
    if pad_size <= 0 {
        return Ok(v.clone());
    }
    let mut width = Vec::with_capacity((ndim * 2) as usize);
    for ax in 0..ndim {
        if ax == 1 {
            width.push(0);
            width.push(pad_size as i32);
        } else {
            width.push(0);
            width.push(0);
        }
    }
    v.pad(&width, 0.0)
}

/// Mamba-2 chunk scan, the mamba_ssm torch fallback ported to MLX. All inputs
/// are f32; `b`/`c` arrive in group form and `a` is already `-exp(A_log)`.
/// Returns (output [B, S, H, D], final_state [B, H, D, NS]).
pub fn chunk_scan(
    x: &MxArray,
    dt: &MxArray,
    a: &MxArray,
    b: &MxArray,
    c: &MxArray,
    d: &MxArray,
    chunk_size: i64,
    initial_states: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let shape = x.shape()?;
    let batch = shape[0];
    let seq = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];
    let b_shape = b.shape()?;
    let num_groups = b_shape[2];
    let state_size = b_shape[3];
    let heads_per_group = num_heads / num_groups;

    let x = x.astype(DType::Float32)?;
    let dt = dt.astype(DType::Float32)?;
    let a = a.astype(DType::Float32)?;
    let b = b.astype(DType::Float32)?;
    let c = c.astype(DType::Float32)?;
    let d = d.astype(DType::Float32)?;

    let b_head = expand_groups(&b, heads_per_group)?; // [B,S,H,NS]
    let c_head = expand_groups(&c, heads_per_group)?;

    let d_residual = d.reshape(&[1, 1, num_heads, 1])?.mul(&x)?; // [B,S,H,D]

    let pad_size = (chunk_size - seq % chunk_size) % chunk_size;
    let padded_seq = seq + pad_size;

    // Discretize on the unpadded tensors, then pad to a full chunk multiple.
    let x = x.mul(&dt.reshape(&[batch, seq, num_heads, 1])?)?;
    let a = a.reshape(&[1, 1, num_heads])?.mul(&dt)?; // [B,S,H]

    let x = pad_time(&x, pad_size, 4)?;
    let a = pad_time(&a, pad_size, 3)?;
    let b_head = pad_time(&b_head, pad_size, 4)?;
    let c_head = pad_time(&c_head, pad_size, 4)?;

    let n_chunks = padded_seq / chunk_size;
    let x_chunks = x.reshape(&[batch, n_chunks, chunk_size, num_heads, head_dim])?;
    let a_chunks = a.reshape(&[batch, n_chunks, chunk_size, num_heads])?;
    let b_chunks = b_head.reshape(&[batch, n_chunks, chunk_size, num_heads, state_size])?;
    let c_chunks = c_head.reshape(&[batch, n_chunks, chunk_size, num_heads, state_size])?;

    let a_perm = a_chunks.transpose(Some(&[0, 3, 1, 2]))?;
    let a_cumsum = a_perm.cumsum(-1)?;

    // 1. Intra-chunk output (diagonal blocks).
    let l = segment_sum(&a_perm, chunk_size)?;
    let l = l.exp()?; // [B,H,N,C,C]
    // G = <C_q, B_k> over the state dim.
    let c_mm = c_chunks.transpose(Some(&[0, 1, 3, 2, 4]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        state_size,
    ])?;
    let b_mm = b_chunks
        .transpose(Some(&[0, 1, 3, 2, 4]))?
        .reshape(&[batch * n_chunks * num_heads, chunk_size, state_size])?
        .transpose(Some(&[0, 2, 1]))?;
    let g = c_mm
        .matmul(&b_mm)?
        .reshape(&[batch, n_chunks, num_heads, chunk_size, chunk_size])?;
    let g = g.transpose(Some(&[0, 1, 3, 4, 2]))?; // [B,N,C,C,H]
    let l_perm = l.transpose(Some(&[0, 2, 3, 4, 1]))?; // [B,N,C,C,H]
    let m = g.mul(&l_perm)?;

    // Y_diag[q,d,h] = sum_k M[q,k,h] * x[k,h,d]
    let m_mm = m.transpose(Some(&[0, 1, 4, 2, 3]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        chunk_size,
    ])?;
    let x_mm = x_chunks.transpose(Some(&[0, 1, 3, 2, 4]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        head_dim,
    ])?;
    let y_diag = m_mm
        .matmul(&x_mm)?
        .reshape(&[batch, n_chunks, num_heads, chunk_size, head_dim])?
        .transpose(Some(&[0, 1, 3, 2, 4]))?; // [B,N,C,H,D]

    // 2. Per-chunk states (B terms).
    let a_last = a_cumsum
        .slice_axis(3, chunk_size - 1, chunk_size)?
        .squeeze(Some(&[3]))?; // [B,H,N]
    let decay_states = a_last.expand_dims(3)?.sub(&a_cumsum)?.exp()?; // [B,H,N,C]
    let decay_states = decay_states.transpose(Some(&[0, 2, 3, 1]))?; // [B,N,C,H]
    let b_decay = b_chunks.mul(&decay_states.expand_dims(4)?)?; // [B,N,C,H,NS]
    let bd_mm = b_decay.transpose(Some(&[0, 1, 3, 2, 4]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        state_size,
    ])?;
    let x2_mm = x_chunks.transpose(Some(&[0, 1, 3, 2, 4]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        head_dim,
    ])?;
    let states = bd_mm
        .transpose(Some(&[0, 2, 1]))?
        .matmul(&x2_mm)?
        .transpose(Some(&[0, 2, 1]))?
        .reshape(&[batch, n_chunks, num_heads, head_dim, state_size])?; // [B,N,H,D,NS]

    // 3. Inter-chunk recurrence (A terms).
    let previous = match initial_states {
        Some(init) => init.expand_dims(1)?, // [B,1,H,D,NS]
        None => MxArray::zeros(
            &[batch, 1, num_heads, head_dim, state_size],
            Some(DType::Float32),
        )?,
    };
    let states_all = MxArray::concatenate(&previous, &states, 1)?; // [B,N+1,H,D,NS]

    // decay_chunk[i,j] = exp(sum_{k=j+1..i} a_k) over chunk boundaries.
    let a_last_padded = a_last.pad(&[0, 0, 0, 0, 1, 0], 0.0)?; // [B,H,N+1] (3D)
    let decay_chunk = segment_sum(&a_last_padded, n_chunks + 1)?
        .exp()?
        .transpose(Some(&[0, 2, 3, 1]))?; // [B,N+1,N+1,H]

    let dc_mm = decay_chunk.transpose(Some(&[0, 3, 1, 2]))?.reshape(&[
        batch * num_heads,
        n_chunks + 1,
        n_chunks + 1,
    ])?;
    let st_mm = states_all.transpose(Some(&[0, 2, 1, 3, 4]))?.reshape(&[
        batch * num_heads,
        n_chunks + 1,
        head_dim * state_size,
    ])?;
    let new_states = dc_mm
        .matmul(&st_mm)?
        .reshape(&[batch, num_heads, n_chunks + 1, head_dim, state_size])?
        .transpose(Some(&[0, 2, 1, 3, 4]))?; // [B,N+1,H,D,NS]

    let states_out = new_states.slice_axis(1, 0, n_chunks)?; // [B,N,H,D,NS]
    let final_state = new_states
        .slice_axis(1, n_chunks, n_chunks + 1)?
        .squeeze(Some(&[1]))?; // [B,H,D,NS]

    // 4. State -> output conversion (C terms).
    let state_decay_out = a_cumsum.exp()?; // [B,H,N,C]
    let c_mm2 = c_chunks.transpose(Some(&[0, 1, 3, 2, 4]))?.reshape(&[
        batch * n_chunks * num_heads,
        chunk_size,
        state_size,
    ])?;
    let st_mm2 = states_out.transpose(Some(&[0, 1, 2, 4, 3]))?.reshape(&[
        batch * n_chunks * num_heads,
        state_size,
        head_dim,
    ])?;
    let c_times_states = c_mm2
        .matmul(&st_mm2)?
        .reshape(&[batch, n_chunks, num_heads, chunk_size, head_dim])?
        .transpose(Some(&[0, 1, 3, 2, 4]))?; // [B,N,C,H,D]
    let sdo = state_decay_out.transpose(Some(&[0, 2, 3, 1]))?; // [B,N,C,H]
    let y_off = c_times_states.mul(&sdo.expand_dims(4)?)?; // [B,N,C,H,D]

    let output = y_diag.add(&y_off)?; // [B,N,C,H,D]
    let output = output.reshape(&[batch, padded_seq, num_heads, head_dim])?;
    let output = if pad_size > 0 {
        output.slice_axis(1, 0, seq)?
    } else {
        output
    };
    let output = output.add(&d_residual)?;

    Ok((output, final_state))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::models::nemotron_h::config::NemotronHConfig;

    /// Tiny mixer config for the unit tests.
    fn tiny_cfg() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_norm_epsilon: 1e-5,
            layers_block_type: vec!["linear_attention".into(), "moe".into()],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 3,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 2,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            intermediate_size: 8,
            moe_shared_expert_intermediate_size: 8,
            eos_token_ids: vec![2],
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
        }
    }

    /// The reference default time-step limit: `(0.0, +inf)`, i.e. no clamp.
    const NO_DT_LIMIT: (f64, f64) = (0.0, f64::INFINITY);

    /// Input amplitude for the dt-clamp regression tests. The SSM term is cubic
    /// in the input while the `D * x` skip is linear, so a small input hides a
    /// clamped dt under the parity tolerance and the assertions go vacuous.
    const SIGNAL: f32 = 4.0;

    fn softplus(v: f32) -> f32 {
        // softplus(x) = max(x,0) + log1p(exp(-|x|))
        v.max(0.0) + (-v.abs()).exp().ln_1p()
    }

    /// Apply a `time_step_limit` pair to a raw `softplus(dt + dt_bias)`, as
    /// `NemotronHMamba2Mixer::forward` does. The oracles must take the limit as
    /// DATA — hardcode a clamp here and no parity assertion can see one.
    fn apply_dt_limit(v: f32, limit: (f64, f64)) -> f32 {
        v.clamp(limit.0 as f32, limit.1 as f32)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn assert_close(actual: &[f32], expected: &[f32], label: &str, tol: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let d = max_abs_diff(actual, expected);
        if d > tol {
            let mut worst = 0usize;
            for i in 0..actual.len() {
                if (actual[i] - expected[i]).abs() > (actual[worst] - expected[worst]).abs() {
                    worst = i;
                }
            }
            panic!(
                "{label}: max |diff| = {d} > {tol} at idx {worst} (actual={} expected={})",
                actual[worst], expected[worst]
            );
        }
    }

    #[test]
    fn chunk_scan_matches_sequential_reference() {
        // Random inputs: B=1, T=5 (padded to 2 chunks of 3), H=2, D=3, G=1, NS=2.
        let h = 2usize;
        let d_dim = 3usize;
        let g = 1usize;
        let ns = 2usize;
        let t = 5usize;
        let chunk = 3i64;

        let x: Vec<f32> = (0..t * h * d_dim)
            .map(|i| ((i as f32) * 0.37) % 2.0 - 1.0)
            .collect();
        let dt: Vec<f32> = (0..t * h)
            .map(|i| 0.05 + 0.2 * ((i as f32) * 0.13) % 1.0)
            .collect();
        let a: Vec<f32> = vec![-0.5, -1.2];
        let b: Vec<f32> = (0..t * g * ns)
            .map(|i| ((i as f32) * 0.61) % 1.0 - 0.5)
            .collect();
        let c: Vec<f32> = (0..t * g * ns)
            .map(|i| ((i as f32) * 0.23) % 1.0 - 0.5)
            .collect();
        let d: Vec<f32> = vec![0.5, -0.3];
        let dt_bias: Vec<f32> = vec![0.1, -0.2];

        let (ref_out, ref_state) = {
            let mut state = vec![0.0f32; h * d_dim * ns];
            let mut out = vec![0.0f32; t * h * d_dim];
            for tt in 0..t {
                for hh in 0..h {
                    let dt_v = apply_dt_limit(softplus(dt[tt * h + hh] + dt_bias[hh]), NO_DT_LIMIT);
                    for dd in 0..d_dim {
                        let xv = x[(tt * h + hh) * d_dim + dd];
                        for ss in 0..ns {
                            let idx = (hh * d_dim + dd) * ns + ss;
                            let bv = b[tt * g * ns + ss];
                            state[idx] = state[idx] * (a[hh] * dt_v).exp() + dt_v * bv * xv;
                        }
                        let mut y = 0.0f32;
                        for ss in 0..ns {
                            y += state[(hh * d_dim + dd) * ns + ss] * c[tt * g * ns + ss];
                        }
                        out[(tt * h + hh) * d_dim + dd] = y + d[hh] * xv;
                    }
                }
            }
            (out, state)
        };

        // The chunk scan receives PRE-PROCESSED dt.
        let dt_proc: Vec<f32> = (0..t * h)
            .map(|i| apply_dt_limit(softplus(dt[i] + dt_bias[i % h]), NO_DT_LIMIT))
            .collect();
        let mx_x = MxArray::from_float32(&x, &[1, t as i64, h as i64, d_dim as i64]).unwrap();
        let mx_dt = MxArray::from_float32(&dt_proc, &[1, t as i64, h as i64]).unwrap();
        let mx_a = MxArray::from_float32(&a, &[h as i64]).unwrap();
        let mx_b = MxArray::from_float32(&b, &[1, t as i64, g as i64, ns as i64]).unwrap();
        let mx_c = MxArray::from_float32(&c, &[1, t as i64, g as i64, ns as i64]).unwrap();
        let mx_d = MxArray::from_float32(&d, &[h as i64]).unwrap();

        // The chunk scan applies x*dt and A*dt internally on the UNSCALED inputs.
        let (got_out, got_state) =
            chunk_scan(&mx_x, &mx_dt, &mx_a, &mx_b, &mx_c, &mx_d, chunk, None)
                .expect("chunk scan runs");
        let got_out = got_out.to_float32().unwrap().to_vec();
        let got_state = got_state.to_float32().unwrap().to_vec();

        // The reference discretizes exactly as the chunk scan does.
        assert_close(&got_out, &ref_out, "chunk scan output vs sequential", 2e-3);
        assert_close(
            &got_state,
            &ref_state,
            "chunk scan final state vs sequential",
            2e-3,
        );
    }

    /// Full-mixer continuity: prefill-all and prefix-prefill + decode must
    /// leave the same SSM state, and the decode output must match a pure-Rust
    /// reference through the whole mixer.
    #[test]
    fn prefill_decode_state_continuity() {
        let cfg = tiny_cfg();
        let mixer = new_mamba_mixer(&cfg).expect("mixer builds");
        let h = cfg.hidden_size as usize;
        let t = 5usize;

        let x: Vec<f32> = (0..t * h)
            .map(|i| (((i as f32) * 0.91) % 2.0) - 1.0)
            .collect();
        let mx_x = MxArray::from_float32(&x, &[1, t as i64, h as i64]).unwrap();

        let (ref_out, ref_state) = mixer_reference(&mixer, &x, t, 1, mixer.dt_limit);

        let mut state = mixer.fresh_state(1).expect("fresh state");
        let full_out = mixer
            .forward(&mx_x, Some(&mut state))
            .expect("prefill runs");
        let full_out = full_out.to_float32().unwrap().to_vec();
        assert_close(&full_out, &ref_out, "prefill output vs reference", 5e-3);
        let prefill_state = state.ssm.to_float32().unwrap().to_vec();
        assert_close(
            &prefill_state,
            &ref_state,
            "prefill state vs reference",
            5e-3,
        );

        let mx_prefix =
            MxArray::from_float32(&x[..(t - 1) * h], &[1, (t - 1) as i64, h as i64]).unwrap();
        let mx_last = MxArray::from_float32(&x[(t - 1) * h..], &[1, 1, h as i64]).unwrap();
        let mut state2 = mixer.fresh_state(1).expect("fresh state 2");
        let _ = mixer
            .forward(&mx_prefix, Some(&mut state2))
            .expect("prefix prefill");
        let decode_out = mixer
            .forward(&mx_last, Some(&mut state2))
            .expect("decode step");
        let decode_out = decode_out.to_float32().unwrap().to_vec();
        let decode_state = state2.ssm.to_float32().unwrap().to_vec();

        assert_close(
            &decode_out,
            &ref_out[(t - 1) * h..],
            "decode output vs reference last token",
            5e-3,
        );
        // Prefill-all and prefix+decode converge to the same SSM state.
        assert_close(&decode_state, &ref_state, "decode state vs reference", 5e-3);
    }

    /// Pure-Rust full-mixer reference: reads the mixer's random weights and
    /// replays in_proj -> conv -> SSM -> gated norm -> out_proj in f32.
    fn mixer_reference(
        mixer: &NemotronHMamba2Mixer,
        x: &[f32],
        t: usize,
        batch: usize,
        dt_limit: (f64, f64),
    ) -> (Vec<f32>, Vec<f32>) {
        let hidden = mixer.in_proj.get_weight().shape_at(1).unwrap() as usize;
        let in_rows = mixer.in_proj.get_weight().shape_at(0).unwrap() as usize;
        let d_inner = mixer.d_inner as usize;
        let conv_dim = mixer.conv1d.get_weight().shape_at(0).unwrap() as usize;
        let num_heads = mixer.num_heads as usize;
        let head_dim = mixer.head_dim as usize;
        let n_groups = mixer.n_groups as usize;
        let ns = mixer.ssm_state_size as usize;
        let hg = num_heads / n_groups;

        let w_in = mixer.in_proj.get_weight().to_float32().unwrap().to_vec(); // [in_rows, hidden]
        let w_out = mixer.out_proj.get_weight().to_float32().unwrap().to_vec();
        let w_conv = mixer.conv1d.get_weight().to_float32().unwrap().to_vec(); // [conv_dim, K, 1]
        // Read the conv bias by forwarding a zero input (output == bias),
        // avoiding Conv1d's private field.
        let conv_bias = {
            // Use a kernel-length input so the conv output length stays >= 1.
            let zero = MxArray::zeros(&[1, 4, conv_dim as i64], Some(DType::Float32)).unwrap();
            Some(
                mixer
                    .conv1d
                    .forward(&zero)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec(),
            )
        };
        let dt_bias = mixer.dt_bias.to_float32().unwrap().to_vec();
        let a_log = mixer.a_log.to_float32().unwrap().to_vec();
        let d = mixer.d.to_float32().unwrap().to_vec();
        let norm_w = mixer.norm_weight.to_float32().unwrap().to_vec();
        let k = mixer.conv_kernel as usize;

        let mut proj = vec![0.0f32; t * in_rows];
        for tt in 0..t {
            for o in 0..in_rows {
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += w_in[o * hidden + i] * x[tt * hidden + i];
                }
                proj[tt * in_rows + o] = acc;
            }
        }
        let mut gate = vec![0.0f32; t * d_inner];
        let mut xbc = vec![0.0f32; t * conv_dim];
        let mut dt = vec![0.0f32; t * num_heads];
        for tt in 0..t {
            for o in 0..d_inner {
                gate[tt * d_inner + o] = proj[tt * in_rows + o];
            }
            for o in 0..conv_dim {
                xbc[tt * conv_dim + o] = proj[tt * in_rows + d_inner + o];
            }
            for o in 0..num_heads {
                dt[tt * num_heads + o] = proj[tt * in_rows + d_inner + conv_dim + o];
            }
        }
        let mut xbc_pad = vec![0.0f32; t * conv_dim + (k - 1) * conv_dim];
        for tt in 0..t {
            for cc in 0..conv_dim {
                xbc_pad[(k - 1 + tt) * conv_dim + cc] = xbc[tt * conv_dim + cc];
            }
        }
        let mut conv_out = vec![0.0f32; t * conv_dim];
        for tt in 0..t {
            for cc in 0..conv_dim {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += w_conv[cc * k + kk] * xbc_pad[(tt + kk) * conv_dim + cc];
                }
                if let Some(b) = &conv_bias {
                    acc += b[cc];
                }
                // SiLU on the conv output, before the SSM split.
                conv_out[tt * conv_dim + cc] = silu(acc);
            }
        }

        let mut x_inner = vec![0.0f32; t * d_inner];
        let mut b_g = vec![0.0f32; t * n_groups * ns];
        let mut c_g = vec![0.0f32; t * n_groups * ns];
        for tt in 0..t {
            for o in 0..d_inner {
                x_inner[tt * d_inner + o] = conv_out[tt * conv_dim + o];
            }
            for o in 0..n_groups * ns {
                b_g[tt * n_groups * ns + o] = conv_out[tt * conv_dim + d_inner + o];
                c_g[tt * n_groups * ns + o] = conv_out[tt * conv_dim + d_inner + n_groups * ns + o];
            }
        }

        let a: Vec<f32> = a_log.iter().map(|v| -v.exp()).collect();
        let mut state = vec![0.0f32; num_heads * head_dim * ns];
        let mut scan = vec![0.0f32; t * num_heads * head_dim];
        for tt in 0..t {
            for hh in 0..num_heads {
                let group = hh / hg;
                let dt_v =
                    apply_dt_limit(softplus(dt[tt * num_heads + hh] + dt_bias[hh]), dt_limit);
                for dd in 0..head_dim {
                    let xv = x_inner[(tt * d_inner) + hh * head_dim + dd];
                    for ss in 0..ns {
                        let idx = (hh * head_dim + dd) * ns + ss;
                        let bv = b_g[(tt * n_groups + group) * ns + ss];
                        state[idx] = state[idx] * (a[hh] * dt_v).exp() + dt_v * bv * xv;
                    }
                    let mut y = 0.0f32;
                    for ss in 0..ns {
                        y += state[(hh * head_dim + dd) * ns + ss]
                            * c_g[(tt * n_groups + group) * ns + ss];
                    }
                    scan[(tt * num_heads + hh) * head_dim + dd] = y + d[hh] * xv;
                }
            }
        }

        let group_size = d_inner / n_groups;
        let mut normed = vec![0.0f32; t * d_inner];
        for tt in 0..t {
            for gg in 0..n_groups {
                let mut sum_sq = 0.0f32;
                for j in 0..group_size {
                    let idx = gg * group_size + j;
                    let v = scan[tt * d_inner + idx] * silu(gate[tt * d_inner + idx]);
                    sum_sq += v * v;
                }
                let rstd = 1.0 / (sum_sq / group_size as f32 + 1e-5).sqrt();
                for j in 0..group_size {
                    let idx = gg * group_size + j;
                    let v = scan[tt * d_inner + idx] * silu(gate[tt * d_inner + idx]);
                    normed[tt * d_inner + idx] = norm_w[idx] * (v * rstd);
                }
            }
        }

        let mut out = vec![0.0f32; t * hidden];
        for tt in 0..t {
            for o in 0..hidden {
                let mut acc = 0.0f32;
                for i in 0..d_inner {
                    acc += w_out[o * d_inner + i] * normed[tt * d_inner + i];
                }
                out[tt * hidden + o] = acc;
            }
        }
        let _ = batch;
        (out, state)
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// Tiny mixer with an explicit per-head `dt_bias` and a NON-ZERO `D` skip.
    /// `D` matters: at `D == 0` the SSM output is near-proportional to `dt` and
    /// the following RMSNorm divides that factor out, hiding a dt clamp.
    fn mixer_with_dt_bias(cfg: &NemotronHConfig, dt_bias: &[f32]) -> NemotronHMamba2Mixer {
        mixer_with_dt_bias_dtype(cfg, dt_bias, DType::Float32)
    }

    /// Cast a fixture weight into the mixer's working dtype (`Float32` is a
    /// no-op, so f32 fixtures stay byte-identical).
    fn as_dtype(v: MxArray, dtype: DType) -> MxArray {
        if dtype == DType::Float32 {
            v
        } else {
            v.astype(dtype).expect("cast fixture weight")
        }
    }

    /// `mixer_with_dt_bias`, with the DENSE weights carried in `dtype`. A dense
    /// `Linear` is a bare matmul and MLX promotes `bf16 x f32 -> f32`, so only
    /// bf16 dense weights reproduce a bf16 stream here.
    ///
    /// `norm_weight` deliberately stays FLOAT32 in every case — that is what
    /// the loader installs from the BF16 checkpoint tensor, and it is the
    /// second half of the promotion `gated_rmsnorm` has to close.
    fn mixer_with_dt_bias_dtype(
        cfg: &NemotronHConfig,
        dt_bias: &[f32],
        dtype: DType,
    ) -> NemotronHMamba2Mixer {
        let mut mixer = new_mamba_mixer(cfg).expect("mixer builds");
        let heads = cfg.mamba_num_heads as i64;
        assert_eq!(dt_bias.len() as i64, heads, "dt_bias must be per-head");

        // Every projection MUST be overwritten deterministically:
        // `new_mamba_mixer` seeds from MLX's PROCESS-GLOBAL RNG, so under the
        // parallel harness the draw depends on what ran first — and the
        // magnitude assertions below are weight-dependent.
        let hidden = cfg.hidden_size as i64;
        let in_rows = cfg.mamba_in_proj_size() as i64;
        let d_inner = cfg.mamba_intermediate_size() as i64;
        let conv_dim = cfg.mamba_conv_dim() as i64;
        let k = cfg.conv_kernel as i64;

        let w_in = det_weights(0, (in_rows * hidden) as usize, 0.5);
        mixer.in_proj = LinearProj::Standard(
            crate::nn::Linear::from_weights(
                &as_dtype(
                    MxArray::from_float32(&w_in, &[in_rows, hidden]).unwrap(),
                    dtype,
                ),
                None,
            )
            .unwrap(),
        );
        let w_out = det_weights(1, (hidden * d_inner) as usize, 0.5);
        mixer.out_proj = LinearProj::Standard(
            crate::nn::Linear::from_weights(
                &as_dtype(
                    MxArray::from_float32(&w_out, &[hidden, d_inner]).unwrap(),
                    dtype,
                ),
                None,
            )
            .unwrap(),
        );
        let w_conv = det_weights(2, (conv_dim * k) as usize, 0.6);
        let b_conv = det_weights(3, conv_dim as usize, 0.2);
        mixer.conv1d = Conv1d::from_weights(
            &as_dtype(
                MxArray::from_float32(&w_conv, &[conv_dim, k, 1]).unwrap(),
                dtype,
            ),
            Some(&as_dtype(
                MxArray::from_float32(&b_conv, &[conv_dim]).unwrap(),
                dtype,
            )),
            None,
            None,
            None,
            Some(conv_dim as u32),
        )
        .unwrap();
        mixer.norm_weight = MxArray::ones(&[d_inner], Some(DType::Float32)).unwrap();
        mixer.a_log = MxArray::from_float32(&[0.0, 0.3], &[heads]).unwrap();
        mixer.dt_bias = MxArray::from_float32(dt_bias, &[heads]).unwrap();
        mixer.d = MxArray::from_float32(&[0.5, -0.3], &[heads]).unwrap();
        mixer
    }

    /// Deterministic pseudo-random weights in `[-scale/2, scale/2)`.
    fn det_weights(stream: u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut v = (i as u64) ^ (stream << 32);
                v = v.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                v ^= v >> 33;
                v = v.wrapping_mul(0xff51_afd7_ed55_8ccd);
                scale * (((v >> 40) % 1024) as f32 / 1024.0 - 0.5)
            })
            .collect()
    }

    /// REGRESSION: `time_step_min` must never bind the runtime clamp. No served
    /// reference does that — only HF's torch fallback.
    ///
    /// Mutation caught: `.clip(Some(self.time_step_min), None)`, or any bound
    /// derived from `time_step_min`, in `forward`. The fixture exaggerates
    /// `time_step_min` to 0.5 so a revived clamp is unmistakable.
    #[test]
    fn time_step_min_does_not_clamp_dt() {
        let mut cfg = tiny_cfg();
        cfg.time_step_min = 0.5;
        assert!(
            cfg.time_step_limit.is_none(),
            "checkpoint declares no limit"
        );

        // softplus(-12) ~= 6.1e-6, far below both 1e-3 and time_step_min.
        let mixer = mixer_with_dt_bias(&cfg, &[-12.0, -12.0]);
        assert_eq!(
            mixer.dt_limit,
            (0.0, f64::INFINITY),
            "an absent time_step_limit must resolve to the unbounded default"
        );

        let h = cfg.hidden_size as usize;
        let t = 4usize;
        let x: Vec<f32> = (0..t * h)
            .map(|i| SIGNAL * ((((i as f32) * 0.91) % 2.0) - 1.0))
            .collect();
        let mx_x = MxArray::from_float32(&x, &[1, t as i64, h as i64]).unwrap();

        let mut state = mixer.fresh_state(1).expect("fresh state");
        let got = mixer
            .forward(&mx_x, Some(&mut state))
            .expect("prefill runs")
            .to_float32()
            .unwrap()
            .to_vec();

        let (ref_unclamped, _) = mixer_reference(&mixer, &x, t, 1, NO_DT_LIMIT);
        assert_close(&got, &ref_unclamped, "mixer vs UNCLAMPED reference", 5e-3);

        // Anti-vacuity: the wrong clamp does move the output materially.
        let (ref_clamped, _) =
            mixer_reference(&mixer, &x, t, 1, (cfg.time_step_min, f64::INFINITY));
        let drift = max_abs_diff(&got, &ref_clamped);
        assert!(
            drift > 1e-2,
            "clamping to time_step_min must change the output (drift {drift})"
        );
    }

    /// The clamp the references DO apply: an explicit `time_step_limit` pair
    /// bounds dt at both ends.
    ///
    /// Mutation caught: dropping `time_step_limit` from the config parse, or
    /// hardcoding `dt_limit` to `(0.0, inf)` in `new_mamba_mixer`.
    #[test]
    fn time_step_limit_clamps_dt_at_both_ends() {
        let mut cfg = tiny_cfg();
        cfg.time_step_limit = Some(vec![0.5, 2.0]);

        // Head 0: softplus(-12) ~= 6.1e-6 -> raised to 0.5.
        // Head 1: softplus(+6)  ~= 6.0    -> lowered to 2.0.
        let mixer = mixer_with_dt_bias(&cfg, &[-12.0, 6.0]);
        assert_eq!(mixer.dt_limit, (0.5, 2.0));

        let h = cfg.hidden_size as usize;
        let t = 4usize;
        let x: Vec<f32> = (0..t * h)
            .map(|i| SIGNAL * ((((i as f32) * 0.57) % 2.0) - 1.0))
            .collect();
        let mx_x = MxArray::from_float32(&x, &[1, t as i64, h as i64]).unwrap();

        let mut state = mixer.fresh_state(1).expect("fresh state");
        let got = mixer
            .forward(&mx_x, Some(&mut state))
            .expect("prefill runs")
            .to_float32()
            .unwrap()
            .to_vec();

        let (ref_clamped, _) = mixer_reference(&mixer, &x, t, 1, (0.5, 2.0));
        assert_close(&got, &ref_clamped, "mixer vs clamped reference", 5e-3);

        let (ref_unclamped, _) = mixer_reference(&mixer, &x, t, 1, NO_DT_LIMIT);
        let drift = max_abs_diff(&got, &ref_unclamped);
        assert!(
            drift > 1e-2,
            "an explicit time_step_limit must change the output (drift {drift})"
        );
    }

    /// The same regression on the recurrent decode path: same `forward`
    /// prologue, `decode_step` instead of the chunk scan.
    #[test]
    fn decode_step_does_not_clamp_dt_to_time_step_min() {
        let mut cfg = tiny_cfg();
        cfg.time_step_min = 0.5;
        let mixer = mixer_with_dt_bias(&cfg, &[-12.0, -12.0]);

        let h = cfg.hidden_size as usize;
        let t = 4usize;
        let x: Vec<f32> = (0..t * h)
            .map(|i| SIGNAL * ((((i as f32) * 0.91) % 2.0) - 1.0))
            .collect();

        // Prefill first, so decode_step runs against a non-zero SSM state.
        let mx_prefix =
            MxArray::from_float32(&x[..(t - 1) * h], &[1, (t - 1) as i64, h as i64]).unwrap();
        let mx_last = MxArray::from_float32(&x[(t - 1) * h..], &[1, 1, h as i64]).unwrap();
        let mut state = mixer.fresh_state(1).expect("fresh state");
        let _ = mixer
            .forward(&mx_prefix, Some(&mut state))
            .expect("prefix prefill");
        let got = mixer
            .forward(&mx_last, Some(&mut state))
            .expect("decode runs")
            .to_float32()
            .unwrap()
            .to_vec();

        let (ref_unclamped, _) = mixer_reference(&mixer, &x, t, 1, NO_DT_LIMIT);
        let last_unclamped = &ref_unclamped[(t - 1) * h..];
        assert_close(&got, last_unclamped, "decode vs UNCLAMPED reference", 5e-3);

        let (ref_clamped, _) =
            mixer_reference(&mixer, &x, t, 1, (cfg.time_step_min, f64::INFINITY));
        let drift = max_abs_diff(&got, &ref_clamped[(t - 1) * h..]);
        assert!(
            drift > 1e-2,
            "clamping to time_step_min must change the decode output (drift {drift})"
        );
    }

    #[test]
    fn decode_single_token_matches_reference() {
        let cfg = tiny_cfg();
        let mixer = new_mamba_mixer(&cfg).expect("mixer builds");
        let h = cfg.hidden_size as usize;
        let x: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.73) % 1.0 - 0.5).collect();
        let mx_x = MxArray::from_float32(&x, &[1, 1, h as i64]).unwrap();

        let mut state = mixer.fresh_state(1).expect("fresh state");
        let out = mixer.forward(&mx_x, Some(&mut state)).expect("decode runs");
        let got = out.to_float32().unwrap().to_vec();
        let (ref_out, _) = mixer_reference(&mixer, &x, 1, 1, mixer.dt_limit);
        assert_close(&got, &ref_out, "single-token decode vs reference", 5e-3);
    }

    /// PINS THE DTYPE CONTRACT at the mixer's exit boundary: `gated_rmsnorm`
    /// must return `out_dtype` whatever its operands are. BOTH fight that — the
    /// scan output is f32 by design and the norm weight is f32 from the
    /// loader's upcast — and MLX promotes, so either one left uncast
    /// re-promotes the product and the f32 escapes into the residual stream.
    ///
    /// Mutations caught, each on its own: dropping `.astype(out_dtype)` on
    /// `normed`, or dropping `weight.astype(out_dtype)`.
    #[test]
    fn gated_rmsnorm_returns_the_requested_io_dtype() {
        let d_inner = 8i64;
        let t = 3i64;
        let xs: Vec<f32> = (0..(t * d_inner))
            .map(|i| ((i as f32) * 0.41) % 2.0 - 1.0)
            .collect();
        let gs: Vec<f32> = (0..(t * d_inner))
            .map(|i| ((i as f32) * 0.73) % 2.0 - 1.0)
            .collect();
        let ws: Vec<f32> = (0..d_inner)
            .map(|i| 0.5 + ((i as f32) * 0.11) % 1.0)
            .collect();

        // Exactly the production dtypes: f32 scan output, f32 norm weight.
        let x = MxArray::from_float32(&xs, &[1, t, d_inner]).unwrap();
        let gate = MxArray::from_float32(&gs, &[1, t, d_inner]).unwrap();
        let weight = MxArray::from_float32(&ws, &[d_inner]).unwrap();
        assert_eq!(x.dtype().unwrap(), DType::Float32);
        assert_eq!(weight.dtype().unwrap(), DType::Float32);

        let f32_out = gated_rmsnorm(&x, &gate, &weight, 1e-5, d_inner, DType::Float32).unwrap();
        assert_eq!(
            f32_out.dtype().unwrap(),
            DType::Float32,
            "an f32 mixer must stay f32 (the cast is a graph no-op there)"
        );

        let bf16_out = gated_rmsnorm(&x, &gate, &weight, 1e-5, d_inner, DType::BFloat16).unwrap();
        assert_eq!(
            bf16_out.dtype().unwrap(),
            DType::BFloat16,
            "the gated norm must return the MIXER's I/O dtype, not the f32 dtype \
             of the scan output or of the upcast norm weight"
        );

        // Only the boundary moved, so the two runs agree to bf16 rounding.
        let a = f32_out.to_float32().unwrap().to_vec();
        let b = bf16_out.to_float32().unwrap().to_vec();
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            scale > 0.1,
            "fixture must produce a non-trivial output (scale {scale})"
        );
        assert_close(&b, &a, "bf16 gated norm vs f32 gated norm", 3e-2 * scale);
    }

    /// The whole mixer must be dtype-transparent: bf16 in -> bf16 out. The only
    /// thing between that reference invariant and an f32 residual stream is the
    /// cast `gated_rmsnorm` applies. The fixture reproduces both production
    /// hazards: an f32 SSM scan and an f32 `norm_weight`.
    ///
    /// Mutation caught: passing `scan_flat.dtype()`, or dropping either cast.
    #[test]
    fn mamba_mixer_is_dtype_transparent() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size as usize;
        let t = 4usize;
        let xs: Vec<f32> = (0..t * h)
            .map(|i| (((i as f32) * 0.91) % 2.0) - 1.0)
            .collect();

        let run = |dtype: DType| -> (DType, Vec<f32>) {
            let mixer = mixer_with_dt_bias_dtype(&cfg, &[0.0, 0.0], dtype);
            assert_eq!(
                mixer.norm_weight.dtype().unwrap(),
                DType::Float32,
                "the fixture must reproduce persistence.rs:539's f32 norm weight, \
                 otherwise the weight half of the promotion is untested"
            );
            let x = as_dtype(
                MxArray::from_float32(&xs, &[1, t as i64, h as i64]).unwrap(),
                dtype,
            );
            let mut state = mixer.fresh_state(1).expect("fresh state");
            let out = mixer.forward(&x, Some(&mut state)).expect("prefill runs");
            (out.dtype().unwrap(), out.to_float32().unwrap().to_vec())
        };

        let (f32_dtype, f32_out) = run(DType::Float32);
        assert_eq!(
            f32_dtype,
            DType::Float32,
            "an f32 stream must stay f32 - the cast must not force half precision"
        );

        let (bf16_dtype, bf16_out) = run(DType::BFloat16);
        assert_eq!(
            bf16_dtype,
            DType::BFloat16,
            "a bf16 stream must leave the mixer in bf16; an f32 return promotes \
             the residual for all 52 layers and forks the flat/paged KV dtypes"
        );

        // Same math, different precision: the two arms must still agree.
        let scale = f32_out.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            scale > 1e-3,
            "fixture must produce a non-trivial output (scale {scale})"
        );
        let drift = max_abs_diff(&bf16_out, &f32_out);
        assert!(
            drift <= 5e-2 * scale,
            "bf16 mixer diverges from the f32 mixer: max |diff| = {drift} (scale {scale})"
        );
    }

    /// The gated-norm epsilon must come from `layer_norm_epsilon`: a sibling
    /// checkpoint declaring a different one would silently normalize wrongly.
    ///
    /// Mutation caught: a literal `1e-5` at the `gated_rmsnorm` call site.
    #[test]
    fn gated_norm_eps_comes_from_the_config() {
        let mut big_eps = tiny_cfg();
        big_eps.layer_norm_epsilon = 4.0;
        let baseline = tiny_cfg();
        assert_eq!(baseline.layer_norm_epsilon, 1e-5);

        let h = baseline.hidden_size as usize;
        let t = 3usize;
        let xs: Vec<f32> = (0..t * h)
            .map(|i| SIGNAL * ((((i as f32) * 0.57) % 2.0) - 1.0))
            .collect();
        let mx = MxArray::from_float32(&xs, &[1, t as i64, h as i64]).unwrap();

        let run = |cfg: &NemotronHConfig| -> Vec<f32> {
            let mixer = mixer_with_dt_bias(cfg, &[0.0, 0.0]);
            assert_eq!(mixer.norm_eps, cfg.layer_norm_epsilon);
            let mut state = mixer.fresh_state(1).expect("fresh state");
            mixer
                .forward(&mx, Some(&mut state))
                .expect("prefill runs")
                .to_float32()
                .unwrap()
                .to_vec()
        };

        let small = run(&baseline);
        let large = run(&big_eps);
        let drift = max_abs_diff(&small, &large);
        assert!(
            drift > 1e-3,
            "a 4.0 epsilon must visibly damp the gated norm (drift {drift})"
        );
    }
}
