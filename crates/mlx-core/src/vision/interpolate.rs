/**
 * Bilinear Interpolation
 *
 * Used for resizing position embeddings and images.
 * GPU-accelerated implementation that avoids CPU roundtrips.
 */
use crate::array::{DType, MxArray};
use napi::bindgen_prelude::*;

/// Bilinear interpolation for 2D spatial data (GPU-accelerated)
///
/// Performs bilinear interpolation on an array whose first two dimensions
/// are spatial (height, width). Supports extra trailing dimensions.
/// All computation stays on GPU for maximum performance.
///
/// # Arguments
/// * `image` - Input array of shape [H, W, ...] where H and W are spatial dimensions
/// * `new_height` - Target height
/// * `new_width` - Target width
///
/// # Returns
/// * Interpolated array of shape [new_height, new_width, ...]
pub fn bilinear_interpolate(image: &MxArray, new_height: i64, new_width: i64) -> Result<MxArray> {
    let shape = image.shape()?;
    if shape.len() < 2 {
        return Err(Error::new(
            Status::InvalidArg,
            "bilinear_interpolate requires at least 2D input",
        ));
    }

    let h_in = shape[0];
    let w_in = shape[1];

    // Handle edge cases
    if new_height == h_in && new_width == w_in {
        return Ok(image.clone());
    }

    if new_height == 1 && new_width == 1 {
        // Just return the center-ish pixel
        let center_h = h_in / 2;
        let center_w = w_in / 2;
        let pixel = image.slice_axis(0, center_h, center_h + 1)?;
        let pixel = pixel.slice_axis(1, center_w, center_w + 1)?;
        return Ok(pixel);
    }

    // Use GPU implementation
    bilinear_interpolate_gpu(image, h_in, w_in, new_height, new_width, &shape)
}

/// GPU-based bilinear interpolation
///
/// Computes bilinear interpolation entirely on GPU using MLX operations.
/// This avoids the CPU roundtrip of the naive implementation.
fn bilinear_interpolate_gpu(
    image: &MxArray,
    h_in: i64,
    w_in: i64,
    new_height: i64,
    new_width: i64,
    shape: &[i64],
) -> Result<MxArray> {
    // Compute source coordinates for each output position
    // Using align_corners=false style: (i + 0.5) * h_in / new_height - 0.5

    // Create output coordinate grids
    let y_out = MxArray::arange(0.0, new_height as f64, Some(1.0), Some(DType::Float32))?; // [new_h]
    let x_out = MxArray::arange(0.0, new_width as f64, Some(1.0), Some(DType::Float32))?; // [new_w]

    // Compute source coordinates using align_corners=false formula
    // y_src = (y_out + 0.5) * h_in / new_height - 0.5
    let y_src = y_out
        .add_scalar(0.5)?
        .mul_scalar(h_in as f64 / new_height as f64)?
        .sub_scalar(0.5)?; // [new_h]

    let x_src = x_out
        .add_scalar(0.5)?
        .mul_scalar(w_in as f64 / new_width as f64)?
        .sub_scalar(0.5)?; // [new_w]

    // Compute floor indices (unclamped for now)
    let y0_f_raw = y_src.floor()?; // [new_h]
    let x0_f_raw = x_src.floor()?; // [new_w]

    // Clamp floor indices to valid range (same as CPU implementation)
    let zero = MxArray::zeros(&[], Some(DType::Float32))?;
    let h_max = MxArray::full(&[], Either::A((h_in - 1) as f64), Some(DType::Float32))?;
    let w_max = MxArray::full(&[], Either::A((w_in - 1) as f64), Some(DType::Float32))?;

    let y0_f = y0_f_raw.maximum(&zero)?.minimum(&h_max)?;
    let x0_f = x0_f_raw.maximum(&zero)?.minimum(&w_max)?;

    // Compute fractional weights using CLAMPED floor indices
    // This matches CPU: (r - rf as f32).clamp(0.0, 1.0) where rf is clamped
    let wy = y_src.sub(&y0_f)?; // [new_h]
    let wx = x_src.sub(&x0_f)?; // [new_w]

    // Clamp weights to [0, 1]
    let zero_f = MxArray::zeros(&[], Some(DType::Float32))?;
    let one_f = MxArray::ones(&[], Some(DType::Float32))?;
    let wy = wy.maximum(&zero_f)?.minimum(&one_f)?; // [new_h]
    let wx = wx.maximum(&zero_f)?.minimum(&one_f)?; // [new_w]

    // Compute ceil indices and clamp
    let y1_f = y0_f.add_scalar(1.0)?.minimum(&h_max)?;
    let x1_f = x0_f.add_scalar(1.0)?.minimum(&w_max)?;

    // Convert to int32 for indexing
    let y0 = y0_f.astype(DType::Int32)?; // [new_h]
    let y1 = y1_f.astype(DType::Int32)?; // [new_h]
    let x0 = x0_f.astype(DType::Int32)?; // [new_w]
    let x1 = x1_f.astype(DType::Int32)?; // [new_w]

    // Flatten the image for efficient gathering: [h*w, ...]
    let extra_dims: Vec<i64> = shape[2..].to_vec();
    let flat_spatial = h_in * w_in;
    let mut flat_shape = vec![flat_spatial];
    flat_shape.extend(&extra_dims);
    let image_flat = image.reshape(&flat_shape)?; // [h*w, ...]

    // Compute linear indices for all 4 corners of each output pixel
    // For position (i, j), we need: y0[i]*w + x0[j], y0[i]*w + x1[j], y1[i]*w + x0[j], y1[i]*w + x1[j]

    // Expand dimensions for broadcasting:
    // y0, y1: [new_h] -> [new_h, 1]
    // x0, x1: [new_w] -> [1, new_w]
    let y0_exp = y0.expand_dims(1)?; // [new_h, 1]
    let y1_exp = y1.expand_dims(1)?; // [new_h, 1]
    let x0_exp = x0.expand_dims(0)?; // [1, new_w]
    let x1_exp = x1.expand_dims(0)?; // [1, new_w]

    // Compute y * w_in for each row index
    let w_in_arr = MxArray::full(&[], Either::A(w_in as f64), Some(DType::Int32))?;
    let y0_times_w = y0_exp.mul(&w_in_arr)?; // [new_h, 1]
    let y1_times_w = y1_exp.mul(&w_in_arr)?; // [new_h, 1]

    // Compute linear indices: idx = y * w + x
    // Broadcasting: [new_h, 1] + [1, new_w] -> [new_h, new_w]
    let idx_00 = y0_times_w.add(&x0_exp)?; // [new_h, new_w]
    let idx_01 = y0_times_w.add(&x1_exp)?; // [new_h, new_w]
    let idx_10 = y1_times_w.add(&x0_exp)?; // [new_h, new_w]
    let idx_11 = y1_times_w.add(&x1_exp)?; // [new_h, new_w]

    // Flatten indices for take operation
    let num_output = new_height * new_width;
    let idx_00_flat = idx_00.reshape(&[num_output])?;
    let idx_01_flat = idx_01.reshape(&[num_output])?;
    let idx_10_flat = idx_10.reshape(&[num_output])?;
    let idx_11_flat = idx_11.reshape(&[num_output])?;

    // Gather the 4 corner values for each output position
    // image_flat: [h*w, ...], indices: [new_h*new_w]
    // Result: [new_h*new_w, ...]
    let v00 = image_flat.take(&idx_00_flat, 0)?;
    let v01 = image_flat.take(&idx_01_flat, 0)?;
    let v10 = image_flat.take(&idx_10_flat, 0)?;
    let v11 = image_flat.take(&idx_11_flat, 0)?;

    // Compute weights for bilinear blending
    // Need to expand weights to broadcast with channel dimension
    // wy: [new_h], wx: [new_w]
    // We need weights shaped [new_h*new_w, 1, 1, ...] to broadcast with [new_h*new_w, ...]

    // First create 2D weight grids
    let wy_exp = wy.expand_dims(1)?; // [new_h, 1]
    let wx_exp = wx.expand_dims(0)?; // [1, new_w]

    // Broadcast to [new_h, new_w]
    let wy_2d = wy_exp.broadcast_to(&[new_height, new_width])?;
    let wx_2d = wx_exp.broadcast_to(&[new_height, new_width])?;

    // Flatten to [new_h*new_w]
    let wy_flat = wy_2d.reshape(&[num_output])?;
    let wx_flat = wx_2d.reshape(&[num_output])?;

    // For extra dimensions, expand the weights
    // v00, v01, v10, v11 have shape [new_h*new_w, ...] where ... are extra dims
    // weights need shape [new_h*new_w, 1, 1, ...] to broadcast
    let num_extra = extra_dims.len();
    let mut wy_final = wy_flat;
    let mut wx_final = wx_flat;

    for _ in 0..num_extra {
        wy_final = wy_final.expand_dims(-1)?;
        wx_final = wx_final.expand_dims(-1)?;
    }

    // Compute 1 - wy and 1 - wx
    let one_minus_wy = wy_final.mul_scalar(-1.0)?.add_scalar(1.0)?;
    let one_minus_wx = wx_final.mul_scalar(-1.0)?.add_scalar(1.0)?;

    // Bilinear blend: result = v00*(1-wy)*(1-wx) + v01*(1-wy)*wx + v10*wy*(1-wx) + v11*wy*wx
    let w00 = one_minus_wy.mul(&one_minus_wx)?;
    let w01 = one_minus_wy.mul(&wx_final)?;
    let w10 = wy_final.mul(&one_minus_wx)?;
    let w11 = wy_final.mul(&wx_final)?;

    let term00 = v00.mul(&w00)?;
    let term01 = v01.mul(&w01)?;
    let term10 = v10.mul(&w10)?;
    let term11 = v11.mul(&w11)?;

    let result_flat = term00.add(&term01)?.add(&term10)?.add(&term11)?;

    // Reshape to final output shape: [new_height, new_width, ...]
    let mut result_shape = vec![new_height, new_width];
    result_shape.extend(&extra_dims);
    result_flat.reshape(&result_shape)
}

/// CPU-based bilinear interpolation (kept for reference/testing)
///
/// This is the original implementation that does GPU->CPU->GPU roundtrip.
/// Kept for testing and comparison purposes.
#[cfg(test)]
fn bilinear_interpolate_cpu(image: &MxArray, new_height: i64, new_width: i64) -> Result<MxArray> {
    let shape = image.shape()?;
    let h_in = shape[0];
    let w_in = shape[1];

    // Compute sampling positions
    let row_positions: Vec<f32> = if new_height == 1 {
        vec![0.0f32]
    } else {
        // align_corners=false style: (i + 0.5) * h_in / new_height - 0.5
        (0..new_height)
            .map(|i| (i as f32 + 0.5) * h_in as f32 / new_height as f32 - 0.5)
            .collect()
    };

    let col_positions: Vec<f32> = if new_width == 1 {
        vec![0.0f32]
    } else {
        (0..new_width)
            .map(|i| (i as f32 + 0.5) * w_in as f32 / new_width as f32 - 0.5)
            .collect()
    };

    // Compute floor and ceil indices
    let row_floor: Vec<i32> = row_positions.iter().map(|&r| r.floor() as i32).collect();
    let col_floor: Vec<i32> = col_positions.iter().map(|&c| c.floor() as i32).collect();

    // Clamp indices to valid range
    let row_floor: Vec<i32> = row_floor
        .iter()
        .map(|&r| r.clamp(0, h_in as i32 - 1))
        .collect();
    let row_ceil: Vec<i32> = row_floor
        .iter()
        .map(|&r| (r + 1).clamp(0, h_in as i32 - 1))
        .collect();
    let col_floor: Vec<i32> = col_floor
        .iter()
        .map(|&c| c.clamp(0, w_in as i32 - 1))
        .collect();
    let col_ceil: Vec<i32> = col_floor
        .iter()
        .map(|&c| (c + 1).clamp(0, w_in as i32 - 1))
        .collect();

    // Compute interpolation weights
    let row_weight: Vec<f32> = row_positions
        .iter()
        .zip(row_floor.iter())
        .map(|(&r, &rf)| (r - rf as f32).clamp(0.0, 1.0))
        .collect();
    let col_weight: Vec<f32> = col_positions
        .iter()
        .zip(col_floor.iter())
        .map(|(&c, &cf)| (c - cf as f32).clamp(0.0, 1.0))
        .collect();

    // Flatten image for easier indexing: [h*w, ...]
    let extra_dims: Vec<i64> = shape[2..].to_vec();
    let flat_spatial = h_in * w_in;
    let mut flat_shape = vec![flat_spatial];
    flat_shape.extend(&extra_dims);
    let image_flat = image.reshape(&flat_shape)?;

    // Build result by iterating over output positions
    let mut result_data: Vec<f32> = Vec::new();

    // Get flat image data (GPU -> CPU transfer)
    let image_data = image_flat.to_float32()?;

    // Calculate extra dimension size
    let extra_size: i64 = extra_dims.iter().product::<i64>().max(1);

    for i in 0..new_height as usize {
        for j in 0..new_width as usize {
            let r0 = row_floor[i] as i64;
            let r1 = row_ceil[i] as i64;
            let c0 = col_floor[j] as i64;
            let c1 = col_ceil[j] as i64;

            let rw = row_weight[i];
            let cw = col_weight[j];

            // Indices into flat image
            let idx_00 = (r0 * w_in + c0) * extra_size;
            let idx_01 = (r0 * w_in + c1) * extra_size;
            let idx_10 = (r1 * w_in + c0) * extra_size;
            let idx_11 = (r1 * w_in + c1) * extra_size;

            // Bilinear blend for each extra dimension element
            for k in 0..extra_size as usize {
                let v00 = image_data[(idx_00 as usize) + k];
                let v01 = image_data[(idx_01 as usize) + k];
                let v10 = image_data[(idx_10 as usize) + k];
                let v11 = image_data[(idx_11 as usize) + k];

                let value = (1.0 - rw) * (1.0 - cw) * v00
                    + (1.0 - rw) * cw * v01
                    + rw * (1.0 - cw) * v10
                    + rw * cw * v11;
                result_data.push(value);
            }
        }
    }

    // Reshape result (CPU -> GPU transfer)
    let mut result_shape = vec![new_height, new_width];
    result_shape.extend(&extra_dims);
    MxArray::from_float32(&result_data, &result_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_same_size() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let image = MxArray::from_float32(&data, &[2, 2]).unwrap();

        let result = bilinear_interpolate(&image, 2, 2).unwrap();
        let result_data = result.to_float32().unwrap();

        assert_eq!(result_data.len(), 4);
        // Should be approximately same values
        for (a, b) in data.iter().zip(result_data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_interpolate_upscale() {
        // 2x2 -> 4x4
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let image = MxArray::from_float32(&data, &[2, 2]).unwrap();

        let result = bilinear_interpolate(&image, 4, 4).unwrap();
        let shape: Vec<i64> = result.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![4, 4]);

        let result_data = result.to_float32().unwrap();
        assert_eq!(result_data.len(), 16);
    }

    #[test]
    fn test_interpolate_downscale() {
        // 4x4 -> 2x2
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let image = MxArray::from_float32(&data, &[4, 4]).unwrap();

        let result = bilinear_interpolate(&image, 2, 2).unwrap();
        let shape: Vec<i64> = result.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn test_interpolate_with_extra_dims() {
        // 2x2x3 (like position embeddings) -> 4x4x3
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let image = MxArray::from_float32(&data, &[2, 2, 3]).unwrap();

        let result = bilinear_interpolate(&image, 4, 4).unwrap();
        let shape: Vec<i64> = result.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![4, 4, 3]);
    }

    #[test]
    fn test_gpu_vs_cpu_correctness() {
        // Test that GPU implementation produces same results as CPU
        // Use a variety of sizes to test edge cases

        // Test case 1: Simple 2x2 -> 3x3
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let image = MxArray::from_float32(&data, &[2, 2]).unwrap();

        let gpu_result = bilinear_interpolate(&image, 3, 3).unwrap();
        let cpu_result = bilinear_interpolate_cpu(&image, 3, 3).unwrap();

        let gpu_data = gpu_result.to_float32().unwrap();
        let cpu_data = cpu_result.to_float32().unwrap();

        assert_eq!(gpu_data.len(), cpu_data.len());
        for (i, (g, c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            // Use 1e-4 tolerance for floating point differences between CPU and GPU
            assert!(
                (g - c).abs() < 1e-4,
                "Mismatch at index {}: GPU={}, CPU={}",
                i,
                g,
                c
            );
        }

        // Test case 2: With extra dimension (like position embeddings)
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let image = MxArray::from_float32(&data, &[2, 3, 4]).unwrap();

        let gpu_result = bilinear_interpolate(&image, 5, 7).unwrap();
        let cpu_result = bilinear_interpolate_cpu(&image, 5, 7).unwrap();

        let gpu_data = gpu_result.to_float32().unwrap();
        let cpu_data = cpu_result.to_float32().unwrap();

        assert_eq!(gpu_data.len(), cpu_data.len());
        for (i, (g, c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "Mismatch at index {}: GPU={}, CPU={}",
                i,
                g,
                c
            );
        }

        // Test case 3: Downscaling
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let image = MxArray::from_float32(&data, &[8, 8]).unwrap();

        let gpu_result = bilinear_interpolate(&image, 3, 3).unwrap();
        let cpu_result = bilinear_interpolate_cpu(&image, 3, 3).unwrap();

        let gpu_data = gpu_result.to_float32().unwrap();
        let cpu_data = cpu_result.to_float32().unwrap();

        assert_eq!(gpu_data.len(), cpu_data.len());
        for (i, (g, c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "Mismatch at index {}: GPU={}, CPU={}",
                i,
                g,
                c
            );
        }
    }

    #[test]
    fn test_interpolate_realistic_pos_embedding_size() {
        // Test with realistic position embedding dimensions
        // e.g., 27x27 grid with 1152 channels (like vision transformers)
        let h = 27;
        let w = 27;
        let channels = 128; // Using smaller channels for test speed
        let data: Vec<f32> = (0..(h * w * channels))
            .map(|i| (i as f32 / 1000.0).sin())
            .collect();
        let image = MxArray::from_float32(&data, &[h as i64, w as i64, channels as i64]).unwrap();

        // Resize to 24x24
        let result = bilinear_interpolate(&image, 24, 24).unwrap();
        let shape: Vec<i64> = result.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![24, 24, channels as i64]);

        // Verify the result is not all zeros or NaN
        let result_data = result.to_float32().unwrap();
        let non_zero = result_data.iter().filter(|&&x| x.abs() > 1e-10).count();
        assert!(
            non_zero > result_data.len() / 2,
            "Result has too many zeros"
        );
        assert!(
            result_data.iter().all(|x| x.is_finite()),
            "Result contains NaN or Inf"
        );
    }
}
