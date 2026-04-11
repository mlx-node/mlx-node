use crate::array::MxArray;
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::nn::{Conv1d, Linear};
use napi::bindgen_prelude::*;

/// ShortConv: gated depthwise Conv1d layer for LFM2.
///
/// Follows `lfm2.py:112-170` (ShortConv class).
///
/// Forward pass:
///   BCx = in_proj(x)                    [B, T, 3*hidden]
///   B, C, x = split(BCx, 3, axis=-1)
///   Bx = B * x
///   conv_out = conv1d(Bx)               (with appropriate padding or cache)
///   y = C * conv_out
///   return out_proj(y)
pub struct ShortConv {
    conv: Conv1d,
    in_proj: Linear,
    out_proj: Linear,
    l_cache: i32,
    hidden_size: i32,
}

impl ShortConv {
    /// Create a new ShortConv layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension
    /// * `l_cache` - Convolution kernel size (typically 3)
    /// * `conv_bias` - Whether to use bias in conv/linear layers
    pub fn new(hidden_size: i32, l_cache: i32, conv_bias: bool) -> Result<Self> {
        let h = hidden_size as u32;

        // Depthwise Conv1d: groups = hidden_size, kernel = l_cache
        // No padding — we handle padding manually (left-pad for prefill, cache for decode)
        let conv = Conv1d::new(
            h,               // in_channels
            h,               // out_channels
            l_cache as u32,  // kernel_size
            None,            // stride (default 1)
            None,            // padding (0 — we do manual padding)
            None,            // dilation (default 1)
            Some(h),         // groups = hidden_size (depthwise)
            Some(conv_bias), // bias
        )?;

        let in_proj = Linear::new(h, 3 * h, Some(conv_bias))?;
        let out_proj = Linear::new(h, h, Some(conv_bias))?;

        Ok(Self {
            conv,
            in_proj,
            out_proj,
            l_cache,
            hidden_size,
        })
    }

    /// Forward pass through the ShortConv layer.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, hidden_size]
    /// * `cache` - Optional ArraysCache (slot 0 holds conv state)
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(&self, x: &MxArray, cache: Option<&mut ArraysCache>) -> Result<MxArray> {
        // 1. Project to 3x hidden
        let bcx = self.in_proj.forward(x)?;

        // 2. Split into B, C, x along last dimension
        let parts = bcx.split(3, Some(-1))?;
        let b_gate = &parts[0];
        let c_gate = &parts[1];
        let x_val = &parts[2];

        // 3. Gated input: Bx = B * x
        let bx = b_gate.mul(x_val)?;

        // 4. Handle padding / caching
        let bx_padded = if let Some(cache) = cache {
            // Decode mode: use cache for conv state
            let state = cache.get(0);
            if let Some(state) = state {
                // Cache has existing state — prepend it
                let bx_with_state = MxArray::concatenate(state, &bx, 1)?;
                // Update cache: keep last (l_cache - 1) positions
                let n_keep = self.l_cache - 1;
                let total_len = bx_with_state.shape_at(1)?;
                let new_state =
                    bx_with_state.slice_axis(1, total_len - n_keep as i64, total_len)?;
                cache.set(0, new_state);
                bx_with_state
            } else {
                // First decode step: init state to zeros
                let batch = bx.shape_at(0)?;
                let n_keep = self.l_cache - 1;
                let state = MxArray::zeros(
                    &[batch, n_keep as i64, self.hidden_size as i64],
                    Some(bx.dtype()?),
                )?;
                let bx_with_state = MxArray::concatenate(&state, &bx, 1)?;
                // Update cache with last (l_cache - 1) positions
                let total_len = bx_with_state.shape_at(1)?;
                let new_state =
                    bx_with_state.slice_axis(1, total_len - n_keep as i64, total_len)?;
                cache.set(0, new_state);
                bx_with_state
            }
        } else {
            // Prefill mode: left-pad with zeros
            // pad_width for [B, T, hidden]: [(0,0), (l_cache-1, 0), (0,0)]
            let pad_amount = self.l_cache - 1;
            bx.pad(&[0, 0, pad_amount, 0, 0, 0], 0.0)?
        };

        // 5. Apply depthwise conv1d
        let conv_out = self.conv.forward(&bx_padded)?;

        // 6. Gated output: y = C * conv_out
        let y = c_gate.mul(&conv_out)?;

        // 7. Output projection
        self.out_proj.forward(&y)
    }

    // ========== Weight setters ==========

    pub fn set_conv_weight(&mut self, w: &MxArray) -> Result<()> {
        self.conv.set_weight(w)
    }

    pub fn set_conv_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.conv.set_bias(b)
    }

    pub fn set_in_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj.set_weight(w)
    }

    pub fn set_in_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.in_proj.set_bias(b)
    }

    pub fn set_out_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.out_proj.set_weight(w)
    }

    pub fn set_out_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.out_proj.set_bias(b)
    }
}
