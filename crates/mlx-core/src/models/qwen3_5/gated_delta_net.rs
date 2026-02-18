use crate::array::MxArray;
use crate::nn::{Activations, Conv1d, Linear};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::gated_delta_update;
use super::rms_norm_gated::RMSNormGated;

/// GatedDeltaNet: Linear attention module using gated delta recurrence.
///
/// This replaces standard attention in most layers of Qwen3.5.
/// Uses depthwise convolution + state-space recurrence instead of softmax attention.
pub struct GatedDeltaNet {
    // Projections
    in_proj_qkvz: Linear, // hidden → key_dim*2 + value_dim*2 (q,k,v,z combined)
    in_proj_ba: Linear,   // hidden → num_v_heads * 2 (b and a combined)
    conv1d: Conv1d,       // depthwise conv, groups = conv_dim
    norm: RMSNormGated,   // per-head norm: weight dim = value_head_dim
    out_proj: Linear,     // value_dim → hidden

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
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            norm,
            out_proj,
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
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Project to qkvz: [B, T, key_dim*2 + value_dim*2]
        let qkvz = self.in_proj_qkvz.forward(x)?;

        // Project to ba: [B, T, num_v_heads*2]
        let ba = self.in_proj_ba.forward(x)?;

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

    // ========== Weight accessors ==========

    pub fn set_in_proj_qkvz_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz.set_weight(w)
    }
    pub fn set_in_proj_ba_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_ba.set_weight(w)
    }
    pub fn set_conv1d_weight(&mut self, w: &MxArray) -> Result<()> {
        self.conv1d.set_weight(w)
    }
    pub fn set_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.norm.set_weight(w)
    }
    pub fn set_out_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.out_proj.set_weight(w)
    }
    pub fn set_dt_bias(&mut self, w: &MxArray) {
        self.dt_bias = w.clone();
    }
    pub fn set_a_log(&mut self, w: &MxArray) {
        self.a_log = w.clone();
    }

    pub fn get_parameters(&self) -> Vec<(&str, MxArray)> {
        vec![
            ("in_proj_qkvz.weight", self.in_proj_qkvz.get_weight()),
            ("in_proj_ba.weight", self.in_proj_ba.get_weight()),
            ("conv1d.weight", self.conv1d.get_weight()),
            ("norm.weight", self.norm.get_weight()),
            ("out_proj.weight", self.out_proj.get_weight()),
            ("dt_bias", self.dt_bias.clone()),
            ("A_log", self.a_log.clone()),
        ]
    }
}

/// RMS normalization without learnable weight (weight=None in Python).
/// Uses mlx_fast_rms_norm with a ones weight vector.
fn rms_norm_no_weight(x: &MxArray, eps: f32) -> Result<MxArray> {
    // Get the last dimension size for the ones weight
    let shape = x.shape()?;
    let last_dim = *shape
        .last()
        .ok_or_else(|| Error::from_reason("empty shape"))?;
    // Use input's dtype to avoid f32 promotion for bf16/f16 models
    let ones = MxArray::ones(&[last_dim], Some(x.dtype()?))?;
    let handle = unsafe { sys::mlx_fast_rms_norm(x.handle.0, ones.handle.0, eps) };
    MxArray::from_handle(handle, "rms_norm_no_weight")
}
