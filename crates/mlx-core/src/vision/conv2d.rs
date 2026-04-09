/**
 * Conv2d Layer
 *
 * 2D Convolution layer for vision transformers.
 * MLX uses NHWC format: [batch, height, width, channels]
 *
 * Note: This is an internal implementation detail used by PatchEmbedding.
 * Not exposed to TypeScript - users interact with high-level VLModel API.
 */
use crate::array::MxArray;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// 2D Convolution layer (internal)
///
/// Applies a 2D convolution over an input signal composed of several input planes.
/// MLX expects weight shape [out_channels, kernel_h, kernel_w, in_channels]
///
/// Note: Currently only supports the patch embedding case where stride == kernel_size
/// and padding == 0. General convolution is not yet implemented.
pub struct Conv2d {
    /// Convolution weights [out_channels, kernel_h, kernel_w, in_channels]
    weight: Arc<MxArray>,
    /// Optional bias [out_channels]
    bias: Option<Arc<MxArray>>,
    /// Stride in (height, width) dimensions
    stride: (u32, u32),
    /// Padding in (height, width) dimensions
    padding: (u32, u32),
    /// Dilation in (height, width) dimensions
    dilation: (u32, u32),
    /// Number of groups for grouped convolution
    groups: u32,
}

impl Conv2d {
    /// Create a new Conv2d layer
    ///
    /// # Arguments
    /// * `weight` - Convolution weights [out_channels, kernel_h, kernel_w, in_channels]
    /// * `bias` - Optional bias [out_channels]
    /// * `stride` - Optional stride as [stride_h, stride_w], default [1, 1]
    /// * `padding` - Optional padding as [pad_h, pad_w], default [0, 0]
    /// * `dilation` - Optional dilation as [dilation_h, dilation_w], default [1, 1]
    /// * `groups` - Optional number of groups, default 1
    pub fn new(
        weight: &MxArray,
        bias: Option<&MxArray>,
        stride: Option<Vec<u32>>,
        padding: Option<Vec<u32>>,
        dilation: Option<Vec<u32>>,
        groups: Option<u32>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(vec![1, 1]);
        let padding = padding.unwrap_or(vec![0, 0]);
        let dilation = dilation.unwrap_or(vec![1, 1]);

        if stride.len() != 2 {
            return Err(Error::new(
                Status::InvalidArg,
                "stride must have 2 elements [stride_h, stride_w]",
            ));
        }
        if padding.len() != 2 {
            return Err(Error::new(
                Status::InvalidArg,
                "padding must have 2 elements [pad_h, pad_w]",
            ));
        }
        if dilation.len() != 2 {
            return Err(Error::new(
                Status::InvalidArg,
                "dilation must have 2 elements [dilation_h, dilation_w]",
            ));
        }

        Ok(Self {
            weight: Arc::new(weight.clone()),
            bias: bias.map(|b| Arc::new(b.clone())),
            stride: (stride[0], stride[1]),
            padding: (padding[0], padding[1]),
            dilation: (dilation[0], dilation[1]),
            groups: groups.unwrap_or(1),
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, height, width, in_channels] (NHWC format)
    ///
    /// # Returns
    /// * Output tensor [batch, out_height, out_width, out_channels]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let output = conv2d_forward(
            input,
            &self.weight,
            self.bias.as_deref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;

        Ok(output)
    }

    /// Get the weight tensor
    pub fn weight(&self) -> MxArray {
        (*self.weight).clone()
    }

    /// Get the bias tensor if present
    pub fn bias(&self) -> Option<MxArray> {
        self.bias.as_ref().map(|b| (**b).clone())
    }

    /// Get the stride
    pub fn get_stride(&self) -> (u32, u32) {
        self.stride
    }

    /// Get the padding
    pub fn get_padding(&self) -> (u32, u32) {
        self.padding
    }
}

/// Standalone Conv2d forward function
///
/// This is the actual convolution implementation using MLX's conv_general.
/// MLX uses NHWC format for inputs and OHWI format for weights.
pub fn conv2d_forward(
    input: &MxArray,
    weight: &MxArray,
    bias: Option<&MxArray>,
    stride: (u32, u32),
    padding: (u32, u32),
    _dilation: (u32, u32),
    _groups: u32,
) -> Result<MxArray> {
    // For now, implement a basic version using matrix operations
    // TODO: Add proper conv2d FFI when available

    // Get input shape [batch, height, width, in_channels]
    let input_shape = input.shape()?;
    if input_shape.len() != 4 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Conv2d input must be 4D [N,H,W,C], got {}D",
                input_shape.len()
            ),
        ));
    }

    // Get weight shape [out_channels, kernel_h, kernel_w, in_channels]
    let weight_shape = weight.shape()?;
    if weight_shape.len() != 4 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Conv2d weight must be 4D [O,kH,kW,I], got {}D",
                weight_shape.len()
            ),
        ));
    }

    let batch = input_shape[0];
    let in_h = input_shape[1];
    let in_w = input_shape[2];
    let _in_c = input_shape[3];

    let out_c = weight_shape[0];
    let k_h = weight_shape[1];
    let k_w = weight_shape[2];
    let _w_in_c = weight_shape[3];

    // Calculate output dimensions
    let out_h = (in_h + 2 * padding.0 as i64 - k_h) / stride.0 as i64 + 1;
    let out_w = (in_w + 2 * padding.1 as i64 - k_w) / stride.1 as i64 + 1;

    // For patch embedding where kernel_size == stride (no overlap),
    // we can use a more efficient implementation
    if k_h == stride.0 as i64 && k_w == stride.1 as i64 && padding.0 == 0 && padding.1 == 0 {
        // Reshape input: [batch, out_h, k_h, out_w, k_w, in_c]
        let input_reshaped = input.reshape(&[batch, out_h, k_h, out_w, k_w, _in_c])?;

        // Transpose to: [batch, out_h, out_w, k_h, k_w, in_c]
        let input_transposed = input_reshaped.transpose(Some(&[0, 1, 3, 2, 4, 5]))?;

        // Flatten patch dimensions: [batch, out_h, out_w, k_h * k_w * in_c]
        let patch_size = k_h * k_w * _in_c;
        let input_flat = input_transposed.reshape(&[batch, out_h, out_w, patch_size])?;

        // Flatten weight: [out_c, k_h * k_w * in_c]
        let weight_flat = weight.reshape(&[out_c, patch_size])?;

        // Transpose weight for matmul: [k_h * k_w * in_c, out_c]
        let weight_t = weight_flat.transpose(None)?;

        // Reshape input for batched matmul: [batch * out_h * out_w, patch_size]
        let input_2d = input_flat.reshape(&[batch * out_h * out_w, patch_size])?;

        // Matrix multiply: [batch * out_h * out_w, out_c]
        let output_2d = input_2d.matmul(&weight_t)?;

        // Reshape back to [batch, out_h, out_w, out_c]
        let mut output = output_2d.reshape(&[batch, out_h, out_w, out_c])?;

        // Add bias if present
        if let Some(b) = bias {
            output = output.add(b)?;
        }

        return Ok(output);
    }

    // General convolution using MLX native conv2d
    unsafe {
        let input_handle = input.handle.0;
        let weight_handle = weight.handle.0;

        let result_handle = mlx_sys::mlx_conv2d(
            input_handle,
            weight_handle,
            stride.0 as i32,
            stride.1 as i32,
            padding.0 as i32,
            padding.1 as i32,
            _dilation.0 as i32,
            _dilation.1 as i32,
            _groups as i32,
        );

        if result_handle.is_null() {
            return Err(Error::new(
                Status::GenericFailure,
                "MLX conv2d returned null",
            ));
        }

        let mut output = MxArray::from_handle(result_handle, "conv2d")?;

        // Add bias if present
        if let Some(b) = bias {
            output = output.add(b)?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_patch_embedding() {
        // Test patch embedding style convolution (stride == kernel_size)
        // Input: [1, 28, 28, 3] - small image
        // Weight: [64, 14, 14, 3] - 14x14 patches -> 64 channels
        let batch = 1i64;
        let h = 28i64;
        let w = 28i64;
        let in_c = 3i64;
        let out_c = 64i64;
        let k = 14i64;

        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let conv = Conv2d::new(
            &weight,
            None,
            Some(vec![14, 14]),
            Some(vec![0, 0]),
            None,
            None,
        )
        .unwrap();
        let output = conv.forward(&input).unwrap();

        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        // Expected: [1, 2, 2, 64] since 28/14 = 2
        assert_eq!(output_shape, vec![1, 2, 2, 64]);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let batch = 1i64;
        let h = 14i64;
        let w = 14i64;
        let in_c = 3i64;
        let out_c = 8i64;
        let k = 14i64;

        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let bias_data: Vec<f32> = (0..out_c as usize).map(|i| i as f32 * 0.1).collect();
        let bias = MxArray::from_float32(&bias_data, &[out_c]).unwrap();

        let conv = Conv2d::new(
            &weight,
            Some(&bias),
            Some(vec![14, 14]),
            Some(vec![0, 0]),
            None,
            None,
        )
        .unwrap();
        let output = conv.forward(&input).unwrap();

        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 1, 1, 8]);
    }

    // ---- General FFI path tests (mlx_conv2d) ----

    #[test]
    fn test_conv2d_general_3x3_stride1_pad1() {
        // Standard 3x3 conv with stride=1, padding=1 (preserves spatial dims)
        // This hits the general FFI path since kernel_size != stride
        let batch = 1i64;
        let h = 4i64;
        let w = 4i64;
        let in_c = 3i64;
        let out_c = 8i64;
        let k = 3i64;

        // Deterministic input: sequential values scaled to [0, 1)
        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        // Deterministic weights: centered around zero
        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let output = conv2d_forward(
            &input,
            &weight,
            None,
            (1, 1), // stride
            (1, 1), // padding
            (1, 1), // dilation
            1,      // groups
        )
        .unwrap();

        // With stride=1, padding=1, 3x3 kernel: out_dim = (4 + 2*1 - 3)/1 + 1 = 4
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 4, 4, 8]);

        // Verify output is not all zeros (FFI actually computed something)
        output.eval();
        let abs_output = output.abs().unwrap();
        let sum = abs_output.sum(None, None).unwrap();
        sum.eval();
        let sum_data: Vec<f32> = sum.to_float32().unwrap().to_vec();
        assert!(sum_data[0] > 0.0, "conv2d output should not be all zeros");
    }

    #[test]
    fn test_conv2d_general_3x3_stride2_pad1() {
        // 3x3 conv with stride=2, padding=1 (downsamples spatial dims by 2x)
        // Used extensively in PP-DocLayoutV3 backbone for downsampling
        let batch = 1i64;
        let h = 8i64;
        let w = 8i64;
        let in_c = 3i64;
        let out_c = 16i64;
        let k = 3i64;

        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let output = conv2d_forward(
            &input,
            &weight,
            None,
            (2, 2), // stride
            (1, 1), // padding
            (1, 1), // dilation
            1,      // groups
        )
        .unwrap();

        // With stride=2, padding=1, 3x3 kernel: out_dim = (8 + 2*1 - 3)/2 + 1 = 4
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 4, 4, 16]);

        // Verify output is not all zeros
        output.eval();
        let abs_output = output.abs().unwrap();
        let sum = abs_output.sum(None, None).unwrap();
        sum.eval();
        let sum_data: Vec<f32> = sum.to_float32().unwrap().to_vec();
        assert!(sum_data[0] > 0.0, "conv2d output should not be all zeros");
    }

    #[test]
    fn test_conv2d_general_1x1_stride1_pad0() {
        // 1x1 pointwise convolution (channel projection)
        // This also hits the general path since kernel_size (1) == stride (1)
        // BUT padding is (0,0), so it actually hits the fast path.
        // To force the general path, we use conv2d_forward directly and
        // verify correctness of this common configuration regardless of path.
        //
        // Note: 1x1 with stride=1 and pad=0 matches the fast path condition
        // (k_h == stride && k_w == stride && padding == 0). We still test it
        // because it validates the output shape and computation for pointwise ops.
        let batch = 1i64;
        let h = 4i64;
        let w = 4i64;
        let in_c = 64i64;
        let out_c = 32i64;
        let k = 1i64;

        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let output = conv2d_forward(
            &input,
            &weight,
            None,
            (1, 1), // stride
            (0, 0), // padding
            (1, 1), // dilation
            1,      // groups
        )
        .unwrap();

        // With stride=1, padding=0, 1x1 kernel: out_dim = (4 + 0 - 1)/1 + 1 = 4
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 4, 4, 32]);

        // Verify output is not all zeros
        output.eval();
        let abs_output = output.abs().unwrap();
        let sum = abs_output.sum(None, None).unwrap();
        sum.eval();
        let sum_data: Vec<f32> = sum.to_float32().unwrap().to_vec();
        assert!(sum_data[0] > 0.0, "conv2d output should not be all zeros");
    }

    #[test]
    fn test_conv2d_general_3x3_stride1_pad1_with_bias() {
        // General path 3x3 conv with bias to verify bias addition works
        // in the FFI path
        let batch = 1i64;
        let h = 4i64;
        let w = 4i64;
        let in_c = 3i64;
        let out_c = 8i64;
        let k = 3i64;

        // Use all-ones input for predictable behavior
        let input_data: Vec<f32> = vec![1.0; (batch * h * w * in_c) as usize];
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        // Use all-zeros weight so conv output is zero, then bias is the only contributor
        let weight_data: Vec<f32> = vec![0.0; (out_c * k * k * in_c) as usize];
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let bias_data: Vec<f32> = (0..out_c as usize).map(|i| (i + 1) as f32).collect();
        let bias = MxArray::from_float32(&bias_data, &[out_c]).unwrap();

        let output = conv2d_forward(
            &input,
            &weight,
            Some(&bias),
            (1, 1), // stride
            (1, 1), // padding
            (1, 1), // dilation
            1,      // groups
        )
        .unwrap();

        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 4, 4, 8]);

        // With zero weights, output = bias broadcast over [1, 4, 4, 8]
        // Each spatial position should have bias values [1, 2, 3, 4, 5, 6, 7, 8]
        output.eval();
        let data: Vec<f32> = output.to_float32().unwrap().to_vec();
        // Check first spatial position has bias values
        for (c, &val) in data.iter().enumerate().take(out_c as usize) {
            assert!(
                (val - (c + 1) as f32).abs() < 1e-5,
                "Expected bias value {} at channel {}, got {}",
                c + 1,
                c,
                val
            );
        }
    }

    #[test]
    fn test_conv2d_general_5x5_stride1_pad2() {
        // 5x5 conv with stride=1, padding=2 (preserves spatial dims)
        // Verifies the general path works with larger kernels
        let batch = 1i64;
        let h = 6i64;
        let w = 6i64;
        let in_c = 3i64;
        let out_c = 4i64;
        let k = 5i64;

        let input_data: Vec<f32> = (0..(batch * h * w * in_c) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[batch, h, w, in_c]).unwrap();

        let weight_data: Vec<f32> = (0..(out_c * k * k * in_c) as usize)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(&weight_data, &[out_c, k, k, in_c]).unwrap();

        let output = conv2d_forward(
            &input,
            &weight,
            None,
            (1, 1), // stride
            (2, 2), // padding
            (1, 1), // dilation
            1,      // groups
        )
        .unwrap();

        // With stride=1, padding=2, 5x5 kernel: out_dim = (6 + 2*2 - 5)/1 + 1 = 6
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 6, 6, 4]);

        // Verify output is not all zeros
        output.eval();
        let abs_output = output.abs().unwrap();
        let sum = abs_output.sum(None, None).unwrap();
        sum.eval();
        let sum_data: Vec<f32> = sum.to_float32().unwrap().to_vec();
        assert!(sum_data[0] > 0.0, "conv2d output should not be all zeros");
    }
}
