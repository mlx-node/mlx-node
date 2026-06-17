use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// 1D Convolution layer.
///
/// Applies a 1D convolution over an input signal composed of several input planes.
/// Used by GatedDeltaNet for depthwise convolution in Qwen3.5.
///
/// Input shape: `[batch, seq_len, in_channels]`
/// Weight shape: `[out_channels, kernel_size, in_channels/groups]`  (MLX convention after sanitization)
/// Output shape: `[batch, seq_len_out, out_channels]`
pub struct Conv1d {
    weight: MxArray,
    bias: Option<MxArray>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
}

impl Conv1d {
    /// Create a new Conv1d layer with random initialization.
    pub fn new(
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: Option<u32>,
        padding: Option<u32>,
        dilation: Option<u32>,
        groups: Option<u32>,
        use_bias: Option<bool>,
    ) -> Result<Self> {
        let groups = groups.unwrap_or(1) as i32;
        if groups <= 0 {
            return Err(Error::from_reason(format!(
                "Conv1d: groups must be > 0, got {}",
                groups
            )));
        }
        let stride = stride.unwrap_or(1) as i32;
        let padding = padding.unwrap_or(0) as i32;
        let dilation = dilation.unwrap_or(1) as i32;

        // Weight shape: [out_channels, kernel_size, in_channels/groups]
        // For depthwise conv (groups == in_channels == out_channels):
        //   weight shape = [out_channels, kernel_size, 1]
        let weight_shape = [
            out_channels as i64,
            kernel_size as i64,
            (in_channels as i64) / (groups as i64),
        ];
        let scale = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let weight = MxArray::random_uniform(&weight_shape, -scale, scale, None)?;

        let bias = if use_bias.unwrap_or(false) {
            Some(MxArray::zeros(&[out_channels as i64], None)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    /// Create a Conv1d layer from pre-loaded weights.
    pub fn from_weights(
        weight: &MxArray,
        bias: Option<&MxArray>,
        stride: Option<u32>,
        padding: Option<u32>,
        dilation: Option<u32>,
        groups: Option<u32>,
    ) -> Result<Self> {
        Ok(Self {
            weight: weight.clone(),
            bias: bias.cloned(),
            stride: stride.unwrap_or(1) as i32,
            padding: padding.unwrap_or(0) as i32,
            dilation: dilation.unwrap_or(1) as i32,
            groups: groups.unwrap_or(1) as i32,
        })
    }

    /// Forward pass: applies 1D convolution.
    ///
    /// Input shape: `[batch, seq_len, in_channels]`
    /// Output shape: `[batch, seq_len_out, out_channels]`
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_conv1d(
                input.handle.0,
                self.weight.handle.0,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        };
        let result = MxArray::from_handle(handle, "conv1d")?;

        if let Some(ref b) = self.bias {
            result.add(b)
        } else {
            Ok(result)
        }
    }

    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.weight = weight.clone();
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Option<&MxArray>) -> Result<()> {
        self.bias = bias.cloned();
        Ok(())
    }
}

impl Clone for Conv1d {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn max_abs_diff(a: &MxArray, b: &MxArray) -> f32 {
        let af = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bf = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        af.as_ref()
            .iter()
            .zip(bf.as_ref().iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// GDN-style depthwise conv: does a windowed conv over
    /// `[conv_state(keep) ++ window(T)]` produce the SAME per-position output as
    /// per-token convs with a rolling conv_state? This is the bit-exactness
    /// assumption the eager-MTP GDN tape replay depends on.
    #[test]
    fn depthwise_conv_windowed_equals_per_token_rolling() {
        let conv_dim = 64i64;
        let kernel = 4i64;
        let keep = kernel - 1;
        let t = 5i64; // window length (> kernel)

        // Depthwise conv weight [conv_dim, kernel, 1], bf16.
        let w = MxArray::random_normal(&[conv_dim, kernel, 1], 0.0, 0.3, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let conv = Conv1d::from_weights(&w, None, Some(1), Some(0), Some(1), Some(conv_dim as u32))
            .unwrap();

        // Initial conv_state [1, keep, conv_dim] and the T-token window.
        let conv_state0 =
            MxArray::random_normal(&[1, keep, conv_dim], 0.0, 0.3, Some(DType::Float32))
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
        let window = MxArray::random_normal(&[1, t, conv_dim], 0.0, 0.3, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // (A) Windowed: conv over [conv_state0 ++ window] → take last T outputs.
        let win_out = {
            let inp = MxArray::concatenate(&conv_state0, &window, 1).unwrap();
            let out = conv.forward(&inp).unwrap();
            let out_len = out.shape_at(1).unwrap();
            let sliced = out.slice_axis(1, out_len - t, out_len).unwrap();
            sliced.eval();
            sliced
        };

        // (B) Per-token rolling: for each token, conv over [rolling_state ++ tok].
        let per_tok_out = {
            let mut rolling = conv_state0.clone();
            let mut outs: Vec<MxArray> = Vec::new();
            for ti in 0..t {
                let tok = window.slice_axis(1, ti, ti + 1).unwrap(); // [1,1,conv_dim]
                let inp = MxArray::concatenate(&rolling, &tok, 1).unwrap(); // [1, keep+1, conv_dim]
                let out = conv.forward(&inp).unwrap(); // [1, 1, conv_dim]
                outs.push(out);
                // roll: new state = last `keep` of [rolling ++ tok]
                let total = inp.shape_at(1).unwrap();
                rolling = inp.slice_axis(1, total - keep, total).unwrap();
            }
            let refs: Vec<&MxArray> = outs.iter().collect();
            let cat = MxArray::concatenate_many(refs, Some(1)).unwrap();
            cat.eval();
            cat
        };

        let d = max_abs_diff(&win_out, &per_tok_out);
        eprintln!("CONV_DIAG windowed_vs_pertoken_max_abs_diff={d:.6e}");
        assert_eq!(
            d, 0.0,
            "windowed depthwise conv must equal per-token rolling conv bit-for-bit \
             (got {d:.6e}) — otherwise GDN tape recorded q/k/v drift from AR per-token"
        );
    }
}
