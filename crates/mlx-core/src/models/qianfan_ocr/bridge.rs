//! MLP Bridge with Pixel Shuffle
//!
//! Connects InternViT vision encoder output [B, 1024, 1024] to Qwen3 LM input [B, 256, 2560].
//!
//! Pipeline:
//! 1. pixel_shuffle_v2: [B, 1024, 1024] -> [B, 256, 4096]
//!    (spatial 2x2 block merging into channel dimension)
//! 2. LayerNorm(4096)
//! 3. Linear(4096, 2560) + GELU
//! 4. Linear(2560, 2560)

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};

use crate::models::pp_doclayout_v3::persistence::get_tensor;

// ============================================================================
// Pixel Shuffle v2
// ============================================================================

/// Pixel shuffle (v2) downsampling that merges 2x2 spatial blocks into channels.
///
/// Matches the Python InternVL implementation exactly (all 4D operations):
///   1. reshape [N,W,H,C] -> [N, W, H*s, C/s]   (merge trailing H into C)
///   2. permute(0,2,1,3)                          (swap spatial axes)
///   3. reshape [N, H*s, W*s, C/s²]              (merge trailing W into C)
///   4. permute(0,2,1,3)                          (v2: spatial transpose)
///
/// Input:  `[B, seq_len, hidden_size]` where `seq_len = H*W` (e.g. 1024 = 32x32)
/// Output: `[B, new_h * new_w, new_c]` where spatial dims are halved and channels 4x
pub fn pixel_shuffle_v2(x: &MxArray, scale_factor: f64) -> Result<MxArray> {
    let shape = x.shape()?;
    let n = shape[0];
    let seq_len = shape[1];
    let c = shape[2];

    let hw = (seq_len as f64).sqrt() as i64;
    if hw * hw != seq_len {
        return Err(Error::from_reason(format!(
            "pixel_shuffle_v2: seq_len ({}) is not a perfect square",
            seq_len
        )));
    }
    let h = hw;
    let w = hw;

    let new_h = (h as f64 * scale_factor) as i64;
    let new_w = (w as f64 * scale_factor) as i64;
    let new_c = (c as f64 / (scale_factor * scale_factor)) as i64;

    // Python: n, w, h, c = x.size()  -- names axis-1 as W, axis-2 as H
    // Reshape to spatial grid: [N, W, H, C]
    let x = x.reshape(&[n, w, h, c])?;

    // Step 1: N,W,H,C -> N, W, H*scale, C/scale
    // Merges trailing H into C (contiguous last two dims)
    let x = x.reshape(&[n, w, new_h, (c as f64 / scale_factor) as i64])?;

    // Step 2: permute(0,2,1,3) -- swap W and new_h
    let x = x.transpose(Some(&[0, 2, 1, 3]))?;
    // [N, new_h, W, C/scale]

    // Step 3: N, new_h, W, C/scale -> N, new_h, new_w, C/scale^2
    // Merges trailing W into C (contiguous last two dims)
    let x = x.reshape(&[n, new_h, new_w, new_c])?;

    // Step 4: v2 permute(0,2,1,3) -- swap the two spatial axes
    let x = x.transpose(Some(&[0, 2, 1, 3]))?;

    // Flatten spatial to sequence: [N, new_h*new_w, new_c]
    x.reshape(&[n, new_h * new_w, new_c])
}

// ============================================================================
// InternVLBridge
// ============================================================================

/// MLP bridge that projects pixel-shuffled vision features to the LM hidden dimension.
///
/// Forward: vision_output [B, 1024, 1024]
///   -> pixel_shuffle_v2 -> [B, 256, 4096]
///   -> LayerNorm         -> [B, 256, 4096]
///   -> Linear(4096, 2560) -> GELU -> Linear(2560, 2560)
///   -> [B, 256, 2560]
pub(crate) struct InternVLBridge {
    ln: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    downsample_ratio: f64,
}

impl InternVLBridge {
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        downsample_ratio: f64,
    ) -> Result<Self> {
        let ln = LayerNorm::from_weights(
            &get_tensor(weights, &format!("{prefix}.ln.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.ln.bias"))?),
            None,
        )?;

        let linear1 = Linear::from_weights(
            &get_tensor(weights, &format!("{prefix}.linear1.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.linear1.bias"))?),
        )?;

        let linear2 = Linear::from_weights(
            &get_tensor(weights, &format!("{prefix}.linear2.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.linear2.bias"))?),
        )?;

        Ok(Self {
            ln,
            linear1,
            linear2,
            downsample_ratio,
        })
    }

    /// Forward: vision_output [B, seq_len, hidden_size] -> [B, new_seq_len, llm_hidden]
    pub fn forward(&self, vision_output: &MxArray) -> Result<MxArray> {
        // 1. Pixel shuffle: [B, 1024, 1024] -> [B, 256, 4096]
        let x = pixel_shuffle_v2(vision_output, self.downsample_ratio)?;

        // 2. LayerNorm
        let x = self.ln.forward(&x)?;

        // 3. Linear1 + GELU
        let x = self.linear1.forward(&x)?;
        let x = Activations::gelu(&x)?;

        // 4. Linear2
        self.linear2.forward(&x)
    }
}

impl Clone for InternVLBridge {
    fn clone(&self) -> Self {
        Self {
            ln: self.ln.clone(),
            linear1: self.linear1.clone(),
            linear2: self.linear2.clone(),
            downsample_ratio: self.downsample_ratio,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_shuffle_v2_output_shape() {
        // Vision encoder output: [B, 1024, 1024] -> [B, 256, 4096]
        let b = 1i64;
        let seq_len = 1024i64; // 32x32 patches
        let hidden = 1024i64;
        let input = MxArray::random_normal(&[b, seq_len, hidden], 0.0, 1.0, None).unwrap();

        let output = pixel_shuffle_v2(&input, 0.5).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        assert_eq!(shape, vec![1, 256, 4096]);
    }

    #[test]
    fn test_pixel_shuffle_v2_small_tensor_shape() {
        // Small case: [1, 4, 4] (2x2 grid, 4 channels) -> [1, 1, 16] with scale=0.5
        let input = MxArray::random_normal(&[1, 4, 4], 0.0, 1.0, None).unwrap();

        let output = pixel_shuffle_v2(&input, 0.5).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        assert_eq!(shape, vec![1, 1, 16]);
    }

    #[test]
    fn test_pixel_shuffle_v2_values() {
        // Verify the spatial rearrangement is correct with known values.
        // Input: [1, 4, 2] -> 2x2 spatial grid with 2 channels
        // After pixel shuffle with scale=0.5: [1, 1, 8]
        //
        // Spatial layout (2x2 grid, 2 channels each):
        //   Position (0,0): [a, b]
        //   Position (0,1): [c, d]
        //   Position (1,0): [e, f]
        //   Position (1,1): [g, h]
        //
        // Flattened input (row-major): [a, b, c, d, e, f, g, h]
        //
        // Python reference (ps_version="v2"):
        //   x = [1, 2, 2, 2] (n, w, h, c)
        //   After reshape [1, 2, 1, 2, 2]: groups along h
        //   After permute [0,2,1,3,4] -> [1, 1, 2, 2, 2]
        //   After reshape [1, 1, 1, 2, 2, 2]
        //   After v2 permute [0,1,3,2,4,5] -> [1, 1, 1, 1, 2, 2] (identity when new_w=1)
        //   wait, new_w = 2*0.5 = 1, so w/new_w = 2
        //   After reshape [1, 1, 1, 8]
        //   After flatten [1, 1, 8]
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = MxArray::from_float32(&values, &[1, 4, 2]).unwrap();

        let output = pixel_shuffle_v2(&input, 0.5).unwrap();
        output.eval();

        let shape: Vec<i64> = output.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, 1, 8]);

        // Extract output values
        let result: Vec<f32> = output.to_float32().unwrap().to_vec();
        assert_eq!(result.len(), 8);

        // Verify no values are lost (all original values should be present)
        let mut sorted = result.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_pixel_shuffle_v2_exact_order_4x4() {
        // Verify pixel shuffle produces the EXACT same output order as the Python
        // InternVL reference. This test would have caught the original bug where
        // a 5D/6D factored decomposition scrambled the spatial arrangement.
        //
        // Python reference (ps_version="v2"):
        //   import torch
        //   x = torch.arange(1, 49, dtype=torch.float32).reshape(1, 16, 3)
        //   # 16 = 4x4 spatial, 3 channels
        //   x = x.reshape(1, 4, 4, 3)  # [N, W, H, C] in Python naming
        //   # Step 1: view(1, 4, 2, 6) -> merge H into C
        //   x = x.view(1, 4, 2, 6)
        //   # Step 2: permute(0,2,1,3) -> [1, 2, 4, 6]
        //   x = x.permute(0, 2, 1, 3).contiguous()
        //   # Step 3: view(1, 2, 2, 12) -> merge W into C
        //   x = x.view(1, 2, 2, 12)
        //   # Step 4 (v2): permute(0,2,1,3) -> [1, 2, 2, 12]
        //   x = x.permute(0, 2, 1, 3).contiguous()
        //   # Flatten: [1, 4, 12]
        //   x = x.reshape(1, 4, 12)
        //   print(x)
        //
        // Expected output (verified by running the Python code):
        //   tensor([[[ 1,  2,  3,  7,  8,  9, 13, 14, 15, 19, 20, 21],
        //            [ 4,  5,  6, 10, 11, 12, 16, 17, 18, 22, 23, 24],
        //            [25, 26, 27, 31, 32, 33, 37, 38, 39, 43, 44, 45],
        //            [28, 29, 30, 34, 35, 36, 40, 41, 42, 46, 47, 48]]])
        let values: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let input = MxArray::from_float32(&values, &[1, 16, 3]).unwrap();

        let output = pixel_shuffle_v2(&input, 0.5).unwrap();
        output.eval();

        let shape: Vec<i64> = output.shape().unwrap().to_vec();
        assert_eq!(
            shape,
            vec![1, 4, 12],
            "4x4 spatial with scale 0.5 -> 2x2 with 12 channels"
        );

        let result: Vec<f32> = output.to_float32().unwrap().to_vec();
        let expected: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0,
        ];
        assert_eq!(
            result, expected,
            "Pixel shuffle output must match Python InternVL reference exactly"
        );
    }

    #[test]
    fn test_pixel_shuffle_v2_batch_dimension() {
        // Verify batch > 1 works
        let b = 3i64;
        let input = MxArray::random_normal(&[b, 16, 8], 0.0, 1.0, None).unwrap();

        let output = pixel_shuffle_v2(&input, 0.5).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        // 16 = 4x4 grid, scale 0.5 -> 2x2 = 4 tokens, channels = 8 * 4 = 32
        assert_eq!(shape, vec![3, 4, 32]);
    }

    /// Build bridge test weights.
    fn make_bridge_weights(prefix: &str, mlp_in: i64, llm_hidden: i64) -> HashMap<String, MxArray> {
        let mut w: HashMap<String, MxArray> = HashMap::new();

        // LayerNorm
        w.insert(
            format!("{prefix}.ln.weight"),
            MxArray::ones(&[mlp_in], None).unwrap(),
        );
        w.insert(
            format!("{prefix}.ln.bias"),
            MxArray::zeros(&[mlp_in], None).unwrap(),
        );

        // Linear1: [llm_hidden, mlp_in] (weight is [out_features, in_features])
        w.insert(
            format!("{prefix}.linear1.weight"),
            MxArray::random_normal(&[llm_hidden, mlp_in], 0.0, 0.02, None).unwrap(),
        );
        w.insert(
            format!("{prefix}.linear1.bias"),
            MxArray::zeros(&[llm_hidden], None).unwrap(),
        );

        // Linear2: [llm_hidden, llm_hidden]
        w.insert(
            format!("{prefix}.linear2.weight"),
            MxArray::random_normal(&[llm_hidden, llm_hidden], 0.0, 0.02, None).unwrap(),
        );
        w.insert(
            format!("{prefix}.linear2.bias"),
            MxArray::zeros(&[llm_hidden], None).unwrap(),
        );

        w
    }

    #[test]
    fn test_bridge_forward_output_shape() {
        // Full bridge: [B, 1024, 1024] -> [B, 256, 2560]
        // mlp_in = 1024 * (1/0.5)^2 = 4096
        let prefix = "bridge";
        let mlp_in = 4096i64;
        let llm_hidden = 2560i64;
        let weights = make_bridge_weights(prefix, mlp_in, llm_hidden);

        let bridge = InternVLBridge::build(&weights, prefix, 0.5).unwrap();

        let b = 1i64;
        let input = MxArray::random_normal(&[b, 1024, 1024], 0.0, 1.0, None).unwrap();

        let output = bridge.forward(&input).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        assert_eq!(shape, vec![1, 256, 2560]);
    }

    #[test]
    fn test_bridge_forward_values_not_zero() {
        let prefix = "bridge";
        let mlp_in = 4096i64;
        let llm_hidden = 2560i64;
        let weights = make_bridge_weights(prefix, mlp_in, llm_hidden);

        let bridge = InternVLBridge::build(&weights, prefix, 0.5).unwrap();

        let input = MxArray::random_normal(&[1, 1024, 1024], 0.0, 1.0, None).unwrap();
        let output = bridge.forward(&input).unwrap();
        output.eval();

        let abs_sum = output.abs().unwrap().sum(None, None).unwrap();
        abs_sum.eval();
        let sum_val: Vec<f32> = abs_sum.to_float32().unwrap().to_vec();
        assert!(sum_val[0] > 0.0, "Bridge output should not be all zeros");
    }
}
