use crate::array::MxArray;
use crate::nn::{Activations, Conv1d, Linear};
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::{GdnKernelTape, gated_delta_update, gated_delta_update_with_tape};
use super::quantized_linear::{LinearProj, QuantizedLinear};
use super::rms_norm_gated::RMSNormGated;

/// Per-GDN-layer tape recorded during the eager MTP verify forward.
///
/// Holds everything the eager MTP rollback replay needs to reconstruct the
/// AR-exact carried GDN state for a layer:
///   * `kernel` — the `(q, k, v, g, beta)` window handed to the per-step
///     recurrence kernel (used to replay the recurrent state at T=1).
///   * `qkv` — the post-mask, pre-conv `[B, T, conv_dim]` activation (used to
///     rebuild the conv state by slicing the accepted prefix).
///   * `conv_kernel_dim` — depthwise conv kernel size; `keep = conv_kernel_dim
///     - 1` is the conv-state window length.
///
/// All array fields are lazy `MxArray` clones (no eval, no copy) so recording
/// stays inside the fused lazy MLX graph.
#[derive(Clone)]
pub(crate) struct GdnLayerTape {
    pub kernel: GdnKernelTape,
    pub qkv: MxArray,
    pub conv_kernel_dim: i32,
}

impl GdnLayerTape {
    /// Replay the accepted prefix into the pre-verify snapshot caches.
    ///
    /// `accepted_steps = accepted_drafts + 1`. Rebuilds BOTH the recurrent
    /// state (per-step T=1 kernel replay from `snapshot_recurrent`, threading
    /// bf16 between calls = AR round-trip) and the conv state (slice the
    /// accepted prefix of the recorded `qkv` onto `snapshot_conv`), then writes
    /// them into the live cache slots (slot 0 = conv_state, slot 1 =
    /// recurrent_state).
    ///
    /// On full accept this is idempotent with what the verify forward already
    /// set the conv state to; on partial accept it correctly trims to the
    /// accepted prefix. Stays inside the lazy graph (no eval).
    pub(crate) fn replay_into(
        &self,
        cache: &mut ArraysCache,
        snapshot_conv: Option<&MxArray>,
        snapshot_recurrent: Option<&MxArray>,
        accepted_steps: usize,
    ) -> Result<()> {
        // --- Recurrent state ---------------------------------------------
        // Start from the pre-verify (bf16) recurrent state. If the snapshot
        // had no recurrent state (cold cache — should not happen at decode
        // time), zero-init to the recorded shapes via the kernel's own
        // zero-state default by re-deriving from `v`.
        let start_state = match snapshot_recurrent {
            Some(s) => s.clone(),
            None => {
                let batch = self.kernel.v.shape_at(0)?;
                let num_v_heads = self.kernel.v.shape_at(2)?;
                let v_dim = self.kernel.v.shape_at(3)?;
                let k_dim = self.kernel.q.shape_at(3)?;
                MxArray::zeros(
                    &[batch, num_v_heads, v_dim, k_dim],
                    Some(self.kernel.v.dtype()?),
                )?
            }
        };
        let new_recurrent = self
            .kernel
            .replay_recurrent_state(&start_state, accepted_steps)?;
        cache.set(1, new_recurrent);

        // --- Conv state --------------------------------------------------
        let keep = (self.conv_kernel_dim - 1) as i64;
        if keep > 0 {
            let conv_dim = self.qkv.shape_at(2)?;
            // Prefix of the recorded qkv covering exactly the accepted steps.
            let qkv_prefix = self.qkv.slice_axis(1, 0, accepted_steps as i64)?;
            // conv_input = snapshot.conv_state ++ qkv_prefix  (axis 1).
            let conv_input = match snapshot_conv {
                Some(state) => MxArray::concatenate(state, &qkv_prefix, 1)?,
                None => {
                    let batch = self.qkv.shape_at(0)?;
                    let zeros = MxArray::zeros(&[batch, keep, conv_dim], Some(self.qkv.dtype()?))?;
                    MxArray::concatenate(&zeros, &qkv_prefix, 1)?
                }
            };
            // Keep the last `keep` timesteps as the new conv_state — mirrors
            // GatedDeltaNet::forward's conv-state update (cache slot 0).
            let total_len = conv_input.shape_at(1)?;
            if total_len >= keep {
                let new_conv_state = conv_input.slice_axis(1, total_len - keep, total_len)?;
                cache.set(0, new_conv_state);
            }
        }
        Ok(())
    }
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
        })
    }

    /// Precompute the stacked `[qkvz; ba]^T` weight once after both in_proj
    /// weights have been loaded. Forward will then use one matmul (x @ wqb_t)
    /// plus two axis-2 slices instead of two matmuls (x @ w_qkvz.T) + (x @ w_ba.T).
    /// Safe to call repeatedly (idempotent).
    ///
    /// Only applies when both in_proj_qkvz and in_proj_ba are non-quantized
    /// Standard linears. Quantized models stay on the unfused 2-matmul
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
        Ok(())
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
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_with_tape(x, mask, cache, use_kernel, None)
    }

    /// Tape-recording variant of [`GatedDeltaNet::forward`].
    ///
    /// When `tape_sink` is `Some`, records the post-mask pre-conv `qkv` plus
    /// the per-step kernel inputs into a [`GdnLayerTape`] for the eager MTP
    /// rollback replay. When `None`, behavior is byte-identical to
    /// [`GatedDeltaNet::forward`]. All recording is by lazy `.clone()` (no
    /// eval), so it stays inside the fused MLX graph.
    pub(crate) fn forward_with_tape(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        mut cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        mut tape_sink: Option<&mut Option<GdnLayerTape>>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // When the stacked weight is available, do one matmul + two slices.
        // MLX_DISABLE_E51_STACKED_GDN_IN_PROJ=1 reverts to the two-matmul path.
        let qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) as i64;
        let ba_dim = (self.num_v_heads * 2) as i64;
        let (qkvz, ba) = if let Some(wqb_t) = &self.in_proj_qkvz_ba_t
            && std::env::var("MLX_DISABLE_E51_STACKED_GDN_IN_PROJ").is_err()
        {
            let combined = x.matmul(wqb_t)?; // [B, T, qkvz_dim + ba_dim]
            let qkvz = combined.slice_axis(2, 0, qkvz_dim)?;
            let ba = combined.slice_axis(2, qkvz_dim, qkvz_dim + ba_dim)?;
            (qkvz, ba)
        } else {
            // Unfused path: two separate matmuls.
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

        // Record the post-mask, pre-conv `qkv` for the eager MTP tape (lazy
        // clone, no eval). The conv-state rebuild on accept slices the accepted
        // prefix of this exact tensor.
        let tape_qkv = tape_sink.as_ref().map(|_| qkv.clone());

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
        let (y, new_state) = if tape_sink.is_some() {
            // Record the per-step kernel inputs into a local sink, then fold
            // them (plus the recorded qkv) into the layer tape below.
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let result = gated_delta_update_with_tape(
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
                Some(&mut kernel_sink),
            )?;
            if let (Some(sink), Some(kernel), Some(qkv)) = (tape_sink.take(), kernel_sink, tape_qkv)
            {
                *sink = Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                });
            }
            result
        } else {
            gated_delta_update(
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
            )?
        };

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
        self.in_proj_qkvz.set_quantized(ql);
    }
    pub fn set_quantized_in_proj_ba(&mut self, ql: QuantizedLinear) {
        self.in_proj_qkvz_ba_t = None;
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
