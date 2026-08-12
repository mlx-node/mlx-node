use crate::array::MxArray;
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
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
    pub(crate) conv: Conv1d,
    pub(crate) in_proj: LinearProj,
    pub(crate) out_proj: LinearProj,
    l_cache: i32,
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

        let in_proj = LinearProj::Standard(Linear::new(h, 3 * h, Some(conv_bias))?);
        let out_proj = LinearProj::Standard(Linear::new(h, h, Some(conv_bias))?);

        Ok(Self {
            conv,
            in_proj,
            out_proj,
            l_cache,
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
        match cache {
            Some(cache) => {
                let state = cache.get(0).cloned();
                let (output, next_state) = self.forward_with_state(x, state.as_ref())?;
                cache.set(0, next_state)?;
                Ok(output)
            }
            None => self.forward_with_state(x, None).map(|(output, _)| output),
        }
    }

    /// Forward with an explicit convolution-state tensor.
    ///
    /// `state`, when present, has shape `[B, l_cache - 1, hidden]`. Returning
    /// the updated state separately lets the continuous-batching executor stack
    /// independent request states into one batch, run the convolution once,
    /// then scatter one row back to each request without sharing mutable caches.
    pub(crate) fn forward_with_state(
        &self,
        x: &MxArray,
        state: Option<&MxArray>,
    ) -> Result<(MxArray, MxArray)> {
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
        let bx_padded = if let Some(state) = state {
            MxArray::concatenate(state, &bx, 1)?
        } else {
            // First token / prefill: left-pad with zeros.
            // pad_width for [B, T, hidden]: [(0,0), (l_cache-1, 0), (0,0)]
            let pad_amount = self.l_cache - 1;
            bx.pad(&[0, 0, pad_amount, 0, 0, 0], 0.0)?
        };

        let n_keep = self.l_cache - 1;
        let total_len = bx_padded.shape_at(1)?;
        let next_state = bx_padded.slice_axis(1, total_len - i64::from(n_keep), total_len)?;

        // 5. Apply depthwise conv1d
        let conv_out = self.conv.forward(&bx_padded)?;

        // 6. Gated output: y = C * conv_out
        let y = c_gate.mul(&conv_out)?;

        // 7. Output projection
        Ok((self.out_proj.forward(&y)?, next_state))
    }

    // ========== Weight setters ==========

    pub fn set_conv_weight(&mut self, w: &MxArray) -> Result<()> {
        self.conv.set_weight(w)
    }

    pub fn set_conv_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.conv.set_bias(b)
    }

    // NOTE: in_proj / out_proj WEIGHTS are loaded via the `*_proj_mut()`
    // accessors below, which expose the mode-aware `LinearProj`. The
    // persistence layer either installs a `QuantizedLinear` backend (affine /
    // mxfp4 / mxfp8 / nvfp4) via `set_quantized`, or sets a dense bf16 weight
    // via `set_weight`. The LAYER biases (additive `.bias`, distinct from the
    // affine quant zero-point `.biases`) keep dedicated setters that dispatch
    // across BOTH arms: a quantized `LinearProj` threads the bias through
    // `QuantizedLinear::set_bias`. The depthwise `conv` weight is never
    // quantized and keeps its dedicated `set_conv_weight`.

    pub fn set_in_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        set_linear_proj_bias(&mut self.in_proj, b)
    }

    pub fn set_out_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        set_linear_proj_bias(&mut self.out_proj, b)
    }

    // ========== Mutable projection accessors ==========
    //
    // Expose the mode-aware `LinearProj`s so the persistence layer can install
    // a quantized backend (affine / mxfp4 / mxfp8 / nvfp4) via `set_quantized`,
    // or a plain bf16 weight via `set_weight`. The depthwise `conv` weight is
    // never quantized and keeps its dedicated `set_conv_weight`.

    pub fn in_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.in_proj
    }

    pub fn out_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.out_proj
    }
}

/// Set the additive layer bias on a `LinearProj`, dispatching across both
/// arms. The `Standard` arm uses `Linear::set_bias` (which copies + shape-
/// checks); the `Quantized` arm threads the bias through
/// `QuantizedLinear::set_bias` (the additive `.bias`, NOT the affine quant
/// `.biases`). mxfp4/mxfp8/nvfp4 projections never ship a `.biases`, but they
/// CAN ship an additive `.bias` (lfm2 `conv_bias=true`), so this must work for
/// every mode.
fn set_linear_proj_bias(proj: &mut LinearProj, b: Option<&MxArray>) -> Result<()> {
    match proj {
        LinearProj::Standard(l) => l.set_bias(b),
        LinearProj::Quantized(ql) => {
            ql.set_bias(b.cloned());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &MxArray, expected: &MxArray, label: &str) {
        let actual_shape = actual.shape().unwrap();
        let expected_shape = expected.shape().unwrap();
        assert_eq!(actual_shape.as_ref(), expected_shape.as_ref(), "{label}");
        let actual = actual.to_float32().unwrap();
        let expected = expected.to_float32().unwrap();
        // MLX may select a different floating-point GEMM tile for M=2 than
        // for two M=1 calls. Use a scale-aware bound that is still far below
        // the deliberately distinct state rows, so a row mix-up fails while
        // harmless reduction-order noise does not depend on global test order.
        let max_excess = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs() - (5e-3 + 1e-3 * b.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_excess <= 0.0,
            "{label} exceeded atol=5e-3, rtol=1e-3 by {max_excess}"
        );
    }

    #[test]
    fn explicit_batched_state_matches_independent_rows() {
        let conv = ShortConv::new(4, 3, false).unwrap();
        let x = MxArray::from_float32(&[0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8], &[2, 1, 4])
            .unwrap();
        let state = MxArray::from_float32(
            &[
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, -0.01, -0.02, -0.03, -0.04, -0.05,
                -0.06, -0.07, -0.08,
            ],
            &[2, 2, 4],
        )
        .unwrap();
        let (batched_output, batched_state) = conv.forward_with_state(&x, Some(&state)).unwrap();

        let mut outputs = Vec::new();
        let mut states = Vec::new();
        for row in 0..2 {
            let row_x = x.slice_axis(0, row, row + 1).unwrap();
            let row_state = state.slice_axis(0, row, row + 1).unwrap();
            let (output, next_state) = conv.forward_with_state(&row_x, Some(&row_state)).unwrap();
            outputs.push(output);
            states.push(next_state);
        }
        let serial_output = MxArray::concatenate_many(outputs.iter().collect(), Some(0)).unwrap();
        let serial_state = MxArray::concatenate_many(states.iter().collect(), Some(0)).unwrap();
        assert_close(&batched_output, &serial_output, "output");
        assert_close(&batched_state, &serial_state, "state");
    }
}
