use crate::array::MxArray;
use crate::nn::{Activations, Conv1d, Linear};
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::gated_delta_update;
use super::int8_gemm;
use super::quantized_linear::{LinearProj, QuantizedLinear};
use super::rms_norm_gated::RMSNormGated;

/// Default minimum `M = batch * seq_len` at which the int8 W8A8 prefill GEMM is
/// worth taking for the GDN `in_proj_qkvz` projection. Mirrors the MLP path
/// (`crate::transformer::mlp`): below this (notably `M == 1` decode) per-token
/// activation-quant overhead dominates and int8 regresses vs bf16, so we fall
/// through to the bf16 stacked/legacy path. Override via the SHARED
/// `MLX_INT8_PREFILL_MIN_M` env (same knob the MLP uses).
const INT8_PREFILL_MIN_M_DEFAULT: i64 = 256;

/// Returns `true` when `MLX_INT8_PREFILL_QKVZ` is set to a truthy value
/// (non-empty, not "0"/"false"). The int8 W8A8 path for the GDN qkvz projection
/// is OFF by default and is kept INDEPENDENT of the MLP's `MLX_INT8_PREFILL`
/// flag so the two can be A/B-attributed separately (and the MLP-only verify
/// stays reproducible).
fn int8_prefill_qkvz_enabled() -> bool {
    match std::env::var("MLX_INT8_PREFILL_QKVZ") {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        }
        Err(_) => false,
    }
}

/// The `M` threshold (`batch * seq_len`) at or above which the int8 qkvz path is
/// taken. Reads the SHARED `MLX_INT8_PREFILL_MIN_M` env (same knob as the MLP),
/// falling back to the default.
fn int8_prefill_min_m() -> i64 {
    std::env::var("MLX_INT8_PREFILL_MIN_M")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|&v| v >= 0)
        .unwrap_or(INT8_PREFILL_MIN_M_DEFAULT)
}

/// GatedDeltaNet: Linear attention module using gated delta recurrence.
///
/// This replaces standard attention in most layers of Qwen3.5.
/// Uses depthwise convolution + state-space recurrence instead of softmax attention.
pub struct GatedDeltaNet {
    // Projections
    in_proj_qkvz: LinearProj, // hidden → key_dim*2 + value_dim*2 (q,k,v,z combined)
    in_proj_ba: LinearProj,   // hidden → num_v_heads * 2 (b and a combined)
    conv1d: Conv1d,           // depthwise conv, groups = conv_dim
    norm: RMSNormGated,       // per-head norm: weight dim = value_head_dim
    out_proj: LinearProj,     // value_dim → hidden

    // Learnable parameters
    dt_bias: MxArray, // [num_v_heads]
    a_log: MxArray,   // [num_v_heads]

    // Dimensions
    num_k_heads: i32,
    num_v_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel_dim: i32,
    /// Pre-stacked `[w_qkvz; w_ba]` transposed to `[hidden, qkvz_dim + ba_dim]`.
    /// Populated by `finalize_in_proj()` after weights are loaded. When present
    /// (and non-quantized), `forward()` does ONE matmul + two slices instead of
    /// two separate matmuls.
    in_proj_qkvz_ba_t: Option<MxArray>,
    /// NA int8 W8A8 prefill (opt-in via `MLX_INT8_PREFILL_QKVZ`): opaque int8
    /// quant of the UN-stacked `in_proj_qkvz` weight `[qkvz_dim, hidden]`
    /// (rows = output channels = `N`, matching `quantize_weight_int8`'s `[N, K]`
    /// contract). Populated by `finalize_in_proj()` ONLY when the flag is truthy
    /// and the projection is a non-quantized Standard linear. `None` keeps the
    /// bf16 stacked/legacy path as the unchanged default.
    ///
    /// Only the qkvz weight is int8'd — `in_proj_ba` (b/a recurrence gates) stays
    /// bf16, so on the int8 path the E51 single-matmul fusion is dropped and we
    /// do TWO matmuls (int8 qkvz + bf16 ba). The int8 qkvz GEMM is faster than
    /// the stacked bf16 matmul and ba is tiny, so this is still a net win.
    qkvz_w_i8: Option<MxArray>,
    /// Per-output-channel f32 scale `[qkvz_dim]` for `qkvz_w_i8`.
    qkvz_s_w: Option<MxArray>,
}

impl GatedDeltaNet {
    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_k_heads = config.linear_num_key_heads;
        let num_v_heads = config.linear_num_value_heads;
        let key_head_dim = config.linear_key_head_dim;
        let value_head_dim = config.linear_value_head_dim;
        let conv_kernel_dim = config.linear_conv_kernel_dim;

        let key_dim = num_k_heads * key_head_dim;
        let value_dim = num_v_heads * value_head_dim;
        // conv_dim = q + k + v channels (NOT key_dim + value_dim)
        let conv_dim = key_dim * 2 + value_dim;

        // Combined projection for q, k, v, z
        // Output: key_dim (q) + key_dim (k) + value_dim (v) + value_dim (z)
        let in_proj_qkvz = Linear::new(
            hidden_size as u32,
            (key_dim * 2 + value_dim * 2) as u32,
            Some(false),
        )?;

        // Combined projection for b and a
        let in_proj_ba = Linear::new(hidden_size as u32, (num_v_heads * 2) as u32, Some(false))?;

        // Depthwise conv1d: groups = conv_dim (each channel has its own filter)
        let conv1d = Conv1d::new(
            conv_dim as u32, // in_channels
            conv_dim as u32, // out_channels
            conv_kernel_dim as u32,
            Some(1),               // stride
            Some(0),               // padding (no padding, we prepend conv_state manually)
            Some(1),               // dilation
            Some(conv_dim as u32), // groups = depthwise
            Some(false),           // no bias
        )?;

        // Norm operates per-head: weight dim = value_head_dim (NOT value_dim)
        let norm = RMSNormGated::new(value_head_dim as u32, Some(config.rms_norm_eps))?;
        let out_proj = Linear::new(value_dim as u32, hidden_size as u32, Some(false))?;

        // Learnable parameters
        let dt_bias = MxArray::ones(&[num_v_heads as i64], None)?;
        let a_log = MxArray::zeros(&[num_v_heads as i64], None)?; // Will be loaded from weights

        Ok(Self {
            in_proj_qkvz: LinearProj::Standard(in_proj_qkvz),
            in_proj_ba: LinearProj::Standard(in_proj_ba),
            conv1d,
            norm,
            out_proj: LinearProj::Standard(out_proj),
            dt_bias,
            a_log,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_dim,
            in_proj_qkvz_ba_t: None,
            qkvz_w_i8: None,
            qkvz_s_w: None,
        })
    }

    /// Precompute the stacked `[qkvz; ba]^T` weight once after both in_proj
    /// weights have been loaded. Forward will then use one matmul (x @ wqb_t)
    /// plus two axis-2 slices instead of two matmuls (x @ w_qkvz.T) + (x @ w_ba.T).
    /// Safe to call repeatedly (idempotent).
    ///
    /// Only applies when both in_proj_qkvz and in_proj_ba are non-quantized
    /// Standard linears. Quantized models continue on the legacy 2-matmul
    /// path (no-op here).
    pub fn finalize_in_proj(&mut self) -> Result<()> {
        match (&self.in_proj_qkvz, &self.in_proj_ba) {
            (LinearProj::Standard(_), LinearProj::Standard(_)) => {}
            _ => return Ok(()),
        }
        let w_qkvz = self.in_proj_qkvz.get_weight(); // [qkvz_dim, hidden]
        let w_ba = self.in_proj_ba.get_weight(); // [ba_dim, hidden]
        let stacked = MxArray::concatenate(&w_qkvz, &w_ba, 0)?; // [qkvz_dim+ba_dim, hidden]
        let stacked_t = stacked.transpose(Some(&[1, 0]))?; // [hidden, qkvz_dim+ba_dim]
        stacked_t.eval();
        self.in_proj_qkvz_ba_t = Some(stacked_t);

        // NA int8 W8A8 prefill (opt-in via MLX_INT8_PREFILL_QKVZ): quantize ONLY
        // the qkvz weight to int8 ONCE at load. `in_proj_ba` stays bf16 (its b/a
        // outputs gate the recurrence and are deliberately excluded from quant;
        // see convert.rs `in_proj_ba.` exclusion), so the int8 path forgoes the
        // E51 single-matmul fusion and does TWO matmuls (int8 qkvz + bf16 ba).
        //
        // LAYOUT: `quantize_weight_int8` expects `[N, K]` with rows = output
        // channels so its `s_w[N]` scale broadcasts onto the GEMM accumulator
        // `acc[M, N]`. The UN-transposed `w_qkvz` is `[qkvz_dim, hidden]` =
        // `[N=qkvz_dim, K=hidden]`, already in `[N, K]`, so we pass it straight in
        // (the stacked `*_t` transposed form is for the bf16 matmul ONLY and must
        // NOT be quantized here). `quantize_weight_int8` ALSO hoists the
        // per-forward transpose, returning the opaque int8 weight already in the
        // `[K, N]` kernel layout (opaque to Rust).
        //
        // SCOPE: int8 prefill is DENSE qwen3_5 ONLY. qwen3_5_moe's decoder layer
        // never constructs/finalizes this GDN path with int8 wiring through the
        // MoE persistence route, so `MLX_INT8_PREFILL_QKVZ` is a silent bf16 no-op
        // on MoE (intended — not wired for MoE; documented, not fixed).
        //
        // MEMORY: with the flag ON, BOTH the bf16 stacked `in_proj_qkvz_ba_t`
        // (above) AND the int8 qkvz weight stay resident — the bf16 stacked form
        // is the per-call `Err`-fallback target in `try_forward_qkvz_int8`. Opt-in
        // trades extra weight memory for prefill speed.
        //
        // VALIDATED REGIME: opt-in, greedy (T=0), bf16 models, prefill M>=256.
        // Sampling / MTP / long-context (>8k) are NOT yet validation-gated, which
        // is why this path is deliberately default-OFF.
        self.qkvz_w_i8 = None;
        self.qkvz_s_w = None;
        if int8_prefill_qkvz_enabled() {
            // Fail-soft: if quant fails (e.g. unsupported shape), leave the
            // fields None so forward() stays on the unchanged bf16 path.
            if let Ok((qkvz_i8, qkvz_s)) = int8_gemm::quantize_weight_int8(&w_qkvz) {
                qkvz_i8.eval();
                qkvz_s.eval();
                self.qkvz_w_i8 = Some(qkvz_i8);
                self.qkvz_s_w = Some(qkvz_s);
            }
        }
        Ok(())
    }

    /// NA int8 W8A8 prefill path for the GDN `in_proj_qkvz` projection.
    ///
    /// Returns:
    ///   * `Ok(Some(qkvz))` — the int8 path ran and produced the qkvz output
    ///     `[B, T, qkvz_dim]` (bf16). The caller computes `ba` separately as
    ///     bf16 (`in_proj_ba` stays full precision).
    ///   * `Ok(None)`       — not eligible (flag off / no int8 weights / `M`
    ///     below threshold / a fail-soft int8-op `Err`); caller uses bf16.
    ///   * `Err`            — only a genuine non-int8 error (reshape/shape).
    ///
    /// Pipeline:
    ///   `x[B,T,hidden]` → `[M, hidden]`
    ///   → `qkvz = int8_w8a8(x, qkvz_w_i8, qkvz_s_w)` `[M, qkvz_dim]`
    ///   → reshape back to `[B, T, qkvz_dim]`.
    ///
    /// The int8 op narrows its result to bf16 internally, so the downstream conv
    /// / recurrence (which expect bf16) are not promoted to f32.
    fn try_forward_qkvz_int8(&self, x: &MxArray) -> Result<Option<MxArray>> {
        // Gate 1: flag + quantized weights present.
        let (Some(qkvz_i8), Some(qkvz_s)) = (&self.qkvz_w_i8, &self.qkvz_s_w) else {
            return Ok(None);
        };
        if !int8_prefill_qkvz_enabled() {
            return Ok(None);
        }

        // Gate 2: M = product of leading dims (everything but the last).
        let shape = x.shape()?;
        let dims: &[i64] = &shape;
        if dims.len() < 2 {
            return Ok(None);
        }
        let hidden = dims[dims.len() - 1];
        let m: i64 = dims[..dims.len() - 1].iter().product();
        // M == 1 (decode) and small prefill regress vs bf16 → fall through.
        if m < int8_prefill_min_m() {
            return Ok(None);
        }

        // Reshape to 2D [M, hidden] for the int8 GEMM.
        let x2d = x.reshape(&[m, hidden])?;

        // int8 W8A8. On Err (e.g. gen<17 / K%16!=0) fall back to bf16.
        let qkvz2d = match int8_gemm::int8_w8a8_matmul(&x2d, qkvz_i8, qkvz_s) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let qkvz_dim = qkvz2d.shape_at(1)?;

        if std::env::var("MLX_INT8_PREFILL_QKVZ_DEBUG").is_ok() {
            eprintln!("[int8-prefill-qkvz] fired: M={m} hidden={hidden} qkvz_dim={qkvz_dim}");
        }

        // Reshape [M, qkvz_dim] back to the original leading dims + qkvz_dim.
        let mut out_shape: Vec<i64> = dims[..dims.len() - 1].to_vec();
        out_shape.push(qkvz_dim);
        let qkvz = qkvz2d.reshape(&out_shape)?;
        Ok(Some(qkvz))
    }

    /// Forward pass for GatedDeltaNet.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, hidden_size]
    /// * `mask` - Optional mask [B, T]
    /// * `cache` - Optional ArraysCache with 2 slots: [conv_state, recurrent_state]
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        mut cache: Option<&mut ArraysCache>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // When the stacked weight is available, do one matmul + two slices.
        // MLX_DISABLE_E51_STACKED_GDN_IN_PROJ=1 reverts to the two-matmul path.
        let qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) as i64;
        let ba_dim = (self.num_v_heads * 2) as i64;
        let (qkvz, ba) = if let Some(qkvz_i8) = self.try_forward_qkvz_int8(x)? {
            // NA int8 W8A8 prefill (opt-in): qkvz via int8 GEMM (bf16 out), ba
            // stays bf16 (recurrence gates excluded from quant). This forgoes
            // the E51 single-matmul fusion ONLY on the int8 path.
            let ba = self.in_proj_ba.forward(x)?;
            (qkvz_i8, ba)
        } else if let Some(wqb_t) = &self.in_proj_qkvz_ba_t
            && std::env::var("MLX_DISABLE_E51_STACKED_GDN_IN_PROJ").is_err()
        {
            let combined = x.matmul(wqb_t)?; // [B, T, qkvz_dim + ba_dim]
            let qkvz = combined.slice_axis(2, 0, qkvz_dim)?;
            let ba = combined.slice_axis(2, qkvz_dim, qkvz_dim + ba_dim)?;
            (qkvz, ba)
        } else {
            // Legacy path: two separate matmuls.
            let qkvz = self.in_proj_qkvz.forward(x)?;
            let ba = self.in_proj_ba.forward(x)?;
            (qkvz, ba)
        };

        // Split ba into b and a: each [B, T, num_v_heads]
        let b = ba.slice_axis(2, 0, self.num_v_heads as i64)?;
        let a = ba.slice_axis(2, self.num_v_heads as i64, (self.num_v_heads * 2) as i64)?;

        // Split qkvz: qkv goes through conv, z bypasses
        // qkv: [B, T, key_dim*2 + value_dim] = [B, T, conv_dim]
        // z: [B, T, value_dim]
        let qkv = qkvz.slice_axis(2, 0, self.conv_dim as i64)?;
        let z = qkvz.slice_axis(
            2,
            self.conv_dim as i64,
            (self.key_dim * 2 + self.value_dim * 2) as i64,
        )?;

        // Apply mask before conv to prevent masked values leaking through convolution
        let qkv = if let Some(m) = mask {
            // m: [B, T] → [B, T, 1] for broadcasting
            let m_3d = m.reshape(&[batch, seq_len, 1])?;
            // Use qkv's dtype to avoid f32 promotion for bf16/f16 models
            m_3d.where_(&qkv, &MxArray::zeros(&[1], Some(qkv.dtype()?))?)?
        } else {
            qkv
        };

        // Handle conv_state: always prepend padding (zeros or cached state)
        let conv_state = if let Some(ref cache) = cache {
            cache.get(0).cloned()
        } else {
            None
        };

        let conv_input = match conv_state {
            Some(state) => {
                // Prepend cached conv_state: [B, kernel-1, conv_dim]
                MxArray::concatenate(&state, &qkv, 1)?
            }
            None => {
                // No cache: prepend zeros of size (kernel_size - 1)
                // Use qkv's dtype to avoid f32 promotion for bf16/f16 models
                let pad_len = (self.conv_kernel_dim - 1) as i64;
                let zeros =
                    MxArray::zeros(&[batch, pad_len, self.conv_dim as i64], Some(qkv.dtype()?))?;
                MxArray::concatenate(&zeros, &qkv, 1)?
            }
        };

        // Update conv_state in cache
        if let Some(cache) = cache.as_deref_mut() {
            // Save last (kernel_size - 1) timesteps as new conv_state
            let total_len = conv_input.shape_at(1)?;
            let keep = (self.conv_kernel_dim - 1) as i64;
            if total_len >= keep {
                let new_conv_state = conv_input.slice_axis(1, total_len - keep, total_len)?;
                cache.set(0, new_conv_state);
            }
        }

        // Conv1d: [B, T_in, conv_dim] → [B, T_out, conv_dim]
        let conv_out = self.conv1d.forward(&conv_input)?;

        // Take last seq_len timesteps (conv may produce more than seq_len if conv_state was prepended)
        let conv_out_len = conv_out.shape_at(1)?;
        let conv_out = if conv_out_len > seq_len {
            conv_out.slice_axis(1, conv_out_len - seq_len, conv_out_len)?
        } else {
            conv_out
        };

        // Apply SiLU activation
        let conv_out = Activations::silu(&conv_out)?;

        // Split into q, k, v
        let q_flat = conv_out.slice_axis(2, 0, self.key_dim as i64)?;
        let k_flat = conv_out.slice_axis(2, self.key_dim as i64, (self.key_dim * 2) as i64)?;
        let v_flat = conv_out.slice_axis(2, (self.key_dim * 2) as i64, self.conv_dim as i64)?;

        // Reshape to head format
        // q, k: [B, T, key_dim] → [B, T, Hk, Dk]
        let q = q_flat.reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        let k = k_flat.reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        // v: [B, T, value_dim] → [B, T, Hv, Dv]
        let v = v_flat.reshape(&[
            batch,
            seq_len,
            self.num_v_heads as i64,
            self.value_head_dim as i64,
        ])?;

        // Apply RMS norm scaling to q and k (matching Python exactly):
        //   inv_scale = head_k_dim^(-0.5)
        //   q = (inv_scale^2) * rms_norm(q, None, 1e-6)
        //   k = inv_scale * rms_norm(k, None, 1e-6)
        let inv_scale = (self.key_head_dim as f64).powf(-0.5);
        let q_normed = rms_norm_no_weight(&q, 1e-6)?;
        let k_normed = rms_norm_no_weight(&k, 1e-6)?;
        let q = q_normed.mul_scalar(inv_scale * inv_scale)?;
        let k = k_normed.mul_scalar(inv_scale)?;

        // Run gated delta recurrence
        let recurrent_state = cache.as_deref().and_then(|c| c.get(1));
        let (y, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b,
            &self.a_log,
            &self.dt_bias,
            recurrent_state,
            mask,
            use_kernel,
        )?;

        // Update recurrent state in cache
        if let Some(cache) = cache {
            cache.set(1, new_state);
        }

        // Reshape z to per-head format: [B, T, value_dim] → [B, T, Hv, Dv]
        let z = z.reshape(&[
            batch,
            seq_len,
            self.num_v_heads as i64,
            self.value_head_dim as i64,
        ])?;

        // Apply RMSNormGated on per-head tensors: [B, T, Hv, Dv]
        // Norm weight is [Dv], operates on last dimension
        let y_normed = self.norm.forward(&y, Some(&z))?;

        // Flatten heads: [B, T, Hv, Dv] → [B, T, value_dim]
        let y_flat = y_normed.reshape(&[batch, seq_len, self.value_dim as i64])?;

        // Output projection
        self.out_proj.forward(&y_flat)
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_in_proj_qkvz_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz_ba_t = None; // invalidate stacked cache
        // NA int8: invalidate the qkvz int8 quant (re-quantized in finalize).
        self.qkvz_w_i8 = None;
        self.qkvz_s_w = None;
        self.in_proj_qkvz.set_weight(w, "in_proj_qkvz")
    }
    pub fn set_in_proj_ba_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_ba.set_weight(w, "in_proj_ba")
    }
    pub fn set_conv1d_weight(&mut self, w: &MxArray) -> Result<()> {
        self.conv1d.set_weight(w)
    }
    pub fn set_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        // norm.weight may be stored as f32 in checkpoints for precision,
        // but must match model dtype to avoid cascading f32 promotion.
        let target_dtype = self.dt_bias.dtype()?;
        let w_dtype = w.dtype()?;
        if w_dtype != target_dtype {
            let casted = w.astype(target_dtype)?;
            self.norm.set_weight(&casted)
        } else {
            self.norm.set_weight(w)
        }
    }
    pub fn set_out_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.out_proj.set_weight(w, "out_proj")
    }
    pub fn set_dt_bias(&mut self, w: &MxArray) {
        self.dt_bias = w.clone();
    }
    pub fn set_a_log(&mut self, w: &MxArray) -> Result<()> {
        // Cast A_log to model dtype (bf16) to avoid f32→bf16 promotion overhead.
        // The precision difference is negligible for inference.
        self.a_log = w.astype(self.dt_bias.dtype()?)?;
        Ok(())
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_in_proj_qkvz(&mut self, ql: QuantizedLinear) {
        self.in_proj_qkvz_ba_t = None;
        // NA int8: a quantized (affine/fp) qkvz can't use the int8 W8A8 path.
        self.qkvz_w_i8 = None;
        self.qkvz_s_w = None;
        self.in_proj_qkvz.set_quantized(ql);
    }
    pub fn set_quantized_in_proj_ba(&mut self, ql: QuantizedLinear) {
        self.in_proj_qkvz_ba_t = None;
        // NA int8: re-quantizing ba invalidates finalize's qkvz int8 quant.
        self.qkvz_w_i8 = None;
        self.qkvz_s_w = None;
        self.in_proj_ba.set_quantized(ql);
    }
    pub fn set_quantized_out_proj(&mut self, ql: QuantizedLinear) {
        self.out_proj.set_quantized(ql);
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_in_proj_qkvz_weight(&self) -> MxArray {
        self.in_proj_qkvz.get_weight()
    }
    pub fn get_in_proj_ba_weight(&self) -> MxArray {
        self.in_proj_ba.get_weight()
    }
    pub fn get_conv1d_weight(&self) -> MxArray {
        self.conv1d.get_weight()
    }
    pub fn get_norm_weight(&self) -> MxArray {
        self.norm.get_weight()
    }
    pub fn get_out_proj_weight(&self) -> MxArray {
        self.out_proj.get_weight()
    }
    pub fn get_dt_bias(&self) -> MxArray {
        self.dt_bias.clone()
    }
    pub fn get_a_log(&self) -> MxArray {
        self.a_log.clone()
    }
}

/// RMS normalization without learnable weight (weight=None in Python).
/// Uses mlx_fast_rms_norm with nullptr weight (C++ handles nullptr → std::nullopt).
fn rms_norm_no_weight(x: &MxArray, eps: f32) -> Result<MxArray> {
    let handle = unsafe { mlx_sys::mlx_fast_rms_norm(x.handle.0, std::ptr::null_mut(), eps) };
    MxArray::from_handle(handle, "rms_norm_no_weight")
}
