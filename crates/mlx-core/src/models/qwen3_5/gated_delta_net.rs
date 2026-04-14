use crate::array::MxArray;
use crate::nn::{Activations, Conv1d, Linear};
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::debug::log_tensor_stats;
use super::gated_delta::gated_delta_update_inner;
use super::quantized_linear::{LinearProj, QuantizedLinear};
use super::rms_norm_gated::RMSNormGated;

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

    // Metadata
    layer_idx: usize,

    // Dimensions
    num_k_heads: i32,
    num_v_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel_dim: i32,
}

impl GatedDeltaNet {
    pub fn new(config: &Qwen3_5Config, layer_idx: usize) -> Result<Self> {
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
            layer_idx,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_dim,
        })
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
        let layer = self.layer_idx;
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // ----------------------------------------------------------------
        // Phase 3a WASM fast path: single FFI call for the pre-recurrence
        // portion of GatedDeltaNet::forward. Replaces ~15 MLX ops with one
        // fused op that builds an identical lazy graph in C++.
        //
        // Byte-identical token IDs are required, so we skip this path when
        // either projection is quantized (the fused C++ op only handles
        // dense weights) or when the native build is running.
        // ----------------------------------------------------------------
        #[cfg(target_family = "wasm")]
        let fused_pre = matches!(self.in_proj_qkvz, LinearProj::Standard(_))
            && matches!(self.in_proj_ba, LinearProj::Standard(_));
        #[cfg(not(target_family = "wasm"))]
        let fused_pre = false;

        let (q, k, v, z, a, b) = if fused_pre {
            #[cfg(target_family = "wasm")]
            {
                use std::ptr;

                let w_qkvz = self.in_proj_qkvz.get_weight();
                let w_ba = self.in_proj_ba.get_weight();
                let w_conv = self.conv1d.get_weight();

                let conv_state_handle = cache
                    .as_deref()
                    .and_then(|c| c.get(0))
                    .map(|s| s.handle.0)
                    .unwrap_or(ptr::null_mut());
                let mask_handle = mask.map(|m| m.handle.0).unwrap_or(ptr::null_mut());

                let mut q_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut k_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut v_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut z_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut a_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut b_out: *mut mlx_sys::mlx_array = ptr::null_mut();
                let mut new_conv_state_out: *mut mlx_sys::mlx_array = ptr::null_mut();

                let rc = unsafe {
                    mlx_sys::mlx_gdn_prefusion_forward(
                        x.handle.0,
                        w_qkvz.handle.0,
                        w_ba.handle.0,
                        w_conv.handle.0,
                        conv_state_handle,
                        mask_handle,
                        self.num_k_heads,
                        self.num_v_heads,
                        self.key_head_dim,
                        self.value_head_dim,
                        self.conv_kernel_dim,
                        1e-6f32,
                        &mut q_out,
                        &mut k_out,
                        &mut v_out,
                        &mut z_out,
                        &mut a_out,
                        &mut b_out,
                        &mut new_conv_state_out,
                    )
                };

                if rc != 0
                    || q_out.is_null()
                    || k_out.is_null()
                    || v_out.is_null()
                    || z_out.is_null()
                    || a_out.is_null()
                    || b_out.is_null()
                    || new_conv_state_out.is_null()
                {
                    return Err(Error::from_reason(
                        "mlx_gdn_prefusion_forward returned an error",
                    ));
                }

                let q = MxArray::from_handle(q_out, "gdn_prefusion_q")?;
                let k = MxArray::from_handle(k_out, "gdn_prefusion_k")?;
                let v = MxArray::from_handle(v_out, "gdn_prefusion_v")?;
                let z = MxArray::from_handle(z_out, "gdn_prefusion_z")?;
                let a = MxArray::from_handle(a_out, "gdn_prefusion_a")?;
                let b = MxArray::from_handle(b_out, "gdn_prefusion_b")?;
                let new_conv_state =
                    MxArray::from_handle(new_conv_state_out, "gdn_prefusion_new_conv_state")?;

                // The fused op already computed the new conv_state; persist it
                // exactly like the per-op path does.
                if let Some(cache) = cache.as_deref_mut() {
                    cache.set(0, new_conv_state);
                }

                log_tensor_stats(&format!("layer.{layer:02}.gdn.q_normed"), &q);
                log_tensor_stats(&format!("layer.{layer:02}.gdn.k_normed"), &k);

                (q, k, v, z, a, b)
            }
            #[cfg(not(target_family = "wasm"))]
            unreachable!()
        } else {
            // Project to qkvz: [B, T, key_dim*2 + value_dim*2]
            let qkvz = self.in_proj_qkvz.forward(x)?;

            // Project to ba: [B, T, num_v_heads*2]
            let ba = self.in_proj_ba.forward(x)?;

            // Split ba into b and a: each [B, T, num_v_heads]
            let b = ba.slice_axis(2, 0, self.num_v_heads as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.ba_b"), &b);
            let a = ba.slice_axis(2, self.num_v_heads as i64, (self.num_v_heads * 2) as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.ba_a"), &a);

            // Split qkvz: qkv goes through conv, z bypasses
            // qkv: [B, T, key_dim*2 + value_dim] = [B, T, conv_dim]
            // z: [B, T, value_dim]
            let qkv = qkvz.slice_axis(2, 0, self.conv_dim as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.qkv"), &qkv);
            let z = qkvz.slice_axis(
                2,
                self.conv_dim as i64,
                (self.key_dim * 2 + self.value_dim * 2) as i64,
            )?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.z"), &z);
            let conv1d_weight = self.conv1d.get_weight();
            log_tensor_stats(
                &format!("layer.{layer:02}.gdn.conv1d_weight"),
                &conv1d_weight,
            );
            log_tensor_stats(&format!("layer.{layer:02}.gdn.dt_bias"), &self.dt_bias);
            log_tensor_stats(&format!("layer.{layer:02}.gdn.a_log"), &self.a_log);

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
                    let zeros = MxArray::zeros(
                        &[batch, pad_len, self.conv_dim as i64],
                        Some(qkv.dtype()?),
                    )?;
                    MxArray::concatenate(&zeros, &qkv, 1)?
                }
            };
            log_tensor_stats(&format!("layer.{layer:02}.gdn.conv_input"), &conv_input);

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
            log_tensor_stats(&format!("layer.{layer:02}.gdn.conv_out"), &conv_out);

            // Take last seq_len timesteps (conv may produce more than seq_len if conv_state was prepended)
            let conv_out_len = conv_out.shape_at(1)?;
            let conv_out = if conv_out_len > seq_len {
                conv_out.slice_axis(1, conv_out_len - seq_len, conv_out_len)?
            } else {
                conv_out
            };

            // Apply SiLU activation
            let conv_out_silu = Activations::silu(&conv_out)?;
            log_tensor_stats(
                &format!("layer.{layer:02}.gdn.conv_out_silu"),
                &conv_out_silu,
            );
            let conv_out = conv_out_silu;

            // Split into q, k, v
            let q_flat = conv_out.slice_axis(2, 0, self.key_dim as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.q_flat"), &q_flat);
            let k_flat = conv_out.slice_axis(2, self.key_dim as i64, (self.key_dim * 2) as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.k_flat"), &k_flat);
            let v_flat =
                conv_out.slice_axis(2, (self.key_dim * 2) as i64, self.conv_dim as i64)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.v_flat"), &v_flat);

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
            log_tensor_stats(&format!("layer.{layer:02}.gdn.q_normed"), &q_normed);
            let k_normed = rms_norm_no_weight(&k, 1e-6)?;
            log_tensor_stats(&format!("layer.{layer:02}.gdn.k_normed"), &k_normed);
            let q = q_normed.mul_scalar(inv_scale * inv_scale)?;
            let k = k_normed.mul_scalar(inv_scale)?;

            (q, k, v, z, a, b)
        };

        // Run gated delta recurrence
        let recurrent_state = cache.as_deref().and_then(|c| c.get(1));
        let (y, new_state) = gated_delta_update_inner(
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
            Some(layer),
        )?;
        log_tensor_stats(&format!("layer.{layer:02}.gdn.y"), &y);
        log_tensor_stats(&format!("layer.{layer:02}.gdn.new_state"), &new_state);

        // Update recurrent state in cache
        if let Some(cache) = cache {
            cache.set(1, new_state);
        }

        // ----------------------------------------------------------------
        // Phase 3b WASM fast path: single FFI call for the post-recurrence
        // tail (reshape z + RMSNormGated + out_proj). Collapses 5 MLX ops
        // (rms_norm + sigmoid + 2 mul + matmul) into one lazy graph in C++.
        //
        // Gated on Standard out_proj only; quantized out_proj falls through
        // to the per-op path. Native build is untouched.
        // ----------------------------------------------------------------
        #[cfg(target_family = "wasm")]
        {
            if matches!(self.out_proj, LinearProj::Standard(_)) {
                use std::ptr;

                let norm_weight = self.norm.get_weight();
                let out_proj_weight = self.out_proj.get_weight();
                let out_proj_bias = self.out_proj.get_bias();
                let bias_handle = out_proj_bias
                    .as_ref()
                    .map(|b| b.handle.0)
                    .unwrap_or(ptr::null_mut());

                // Match the per-op fallback exactly: use RMSNormGated's
                // actual eps (populated from config.rms_norm_eps), not a
                // hardcoded constant that would drift on config changes.
                let rms_eps = self.norm.eps();
                let intermediate_size = self.value_dim;

                let mut y_out_handle: *mut mlx_sys::mlx_array = ptr::null_mut();

                let rc = unsafe {
                    mlx_sys::mlx_gdn_postfusion_forward(
                        y.handle.0,
                        z.handle.0,
                        norm_weight.handle.0,
                        out_proj_weight.handle.0,
                        bias_handle,
                        rms_eps,
                        batch as i32,
                        seq_len as i32,
                        self.num_v_heads,
                        self.value_head_dim,
                        intermediate_size,
                        &mut y_out_handle,
                    )
                };

                if rc != 0 || y_out_handle.is_null() {
                    return Err(Error::from_reason(
                        "mlx_gdn_postfusion_forward returned an error",
                    ));
                }

                let out = MxArray::from_handle(y_out_handle, "gdn_postfusion_out")?;
                log_tensor_stats(&format!("layer.{layer:02}.gdn.out"), &out);
                return Ok(out);
            }
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
        log_tensor_stats(&format!("layer.{layer:02}.gdn.y_normed"), &y_normed);

        // Flatten heads: [B, T, Hv, Dv] → [B, T, value_dim]
        let y_flat = y_normed.reshape(&[batch, seq_len, self.value_dim as i64])?;
        log_tensor_stats(&format!("layer.{layer:02}.gdn.y_flat"), &y_flat);

        // Output projection
        let out = self.out_proj.forward(&y_flat)?;
        log_tensor_stats(&format!("layer.{layer:02}.gdn.out"), &out);
        Ok(out)
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_in_proj_qkvz_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz.set_weight(w, "in_proj_qkvz")
    }
    pub fn set_in_proj_ba_weight(&mut self, w: &MxArray) -> Result<()> {
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
        self.in_proj_qkvz.set_quantized(ql);
    }
    pub fn set_quantized_in_proj_ba(&mut self, ql: QuantizedLinear) {
        self.in_proj_ba.set_quantized(ql);
    }
    pub fn set_quantized_out_proj(&mut self, ql: QuantizedLinear) {
        self.out_proj.set_quantized(ql);
    }

    /// Whether any mode-aware projection in this GDN block is quantized.
    /// Convolution, norms, and recurrent parameters are always dense.
    pub fn is_quantized(&self) -> bool {
        self.in_proj_qkvz.is_quantized()
            || self.in_proj_ba.is_quantized()
            || self.out_proj.is_quantized()
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
