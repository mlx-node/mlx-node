//! KV Scale calibration for FP8 quantization
//!
//! This module provides scale calibration for FP8 KV cache quantization.
//! Proper scale calibration is critical for FP8 to work well - without it,
//! values will either overflow (scale too small) or lose precision (scale too large).
//!
//! ## FP8 E4M3 Format
//! - 1 sign bit, 4 exponent bits, 3 mantissa bits
//! - Range: [-448, 448] with special values for inf/nan
//! - We use 448.0 as the max representable value for scale computation
//!
//! ## Calibration Modes
//! - **One-shot**: Compute scales from a single calibration batch
//! - **Running EMA**: Exponential moving average for online calibration

use std::collections::HashMap;

/// Maximum representable value in FP8 E4M3 format
const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to prevent division by zero
const MIN_SCALE: f32 = 1e-12;

/// Default EMA smoothing factor
const DEFAULT_ALPHA: f32 = 0.1;

/// KV scale manager for FP8 quantization
///
/// Manages per-layer quantization scales for keys and values.
/// Scales are computed as: scale = FP8_MAX / max(abs(tensor))
///
/// During inference:
/// - Quantization: fp8_value = fp32_value * scale
/// - Dequantization: fp32_value = fp8_value / scale
#[derive(Debug, Clone)]
pub struct KvScaleManager {
    /// Per-layer key scales
    k_scales: HashMap<u32, f32>,

    /// Per-layer value scales
    v_scales: HashMap<u32, f32>,

    /// Number of layers
    num_layers: u32,

    /// EMA smoothing factor for running calibration
    alpha: f32,

    /// Whether calibration has been performed
    calibrated: bool,

    /// Running max values for EMA (per layer)
    k_running_max: HashMap<u32, f32>,
    v_running_max: HashMap<u32, f32>,
}

impl KvScaleManager {
    /// Create a new KV scale manager
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    pub fn new(num_layers: u32) -> Self {
        Self {
            k_scales: HashMap::new(),
            v_scales: HashMap::new(),
            num_layers,
            alpha: DEFAULT_ALPHA,
            calibrated: false,
            k_running_max: HashMap::new(),
            v_running_max: HashMap::new(),
        }
    }

    /// Create with custom EMA alpha
    pub fn with_alpha(num_layers: u32, alpha: f32) -> Self {
        let mut manager = Self::new(num_layers);
        manager.alpha = alpha.clamp(0.0, 1.0);
        manager
    }

    /// Initialize all scales to 1.0 (no quantization effect)
    pub fn init_default_scales(&mut self) {
        for layer_idx in 0..self.num_layers {
            self.k_scales.insert(layer_idx, 1.0);
            self.v_scales.insert(layer_idx, 1.0);
        }
    }

    /// Get the key scale for a layer
    pub fn k_scale(&self, layer_idx: u32) -> f32 {
        *self.k_scales.get(&layer_idx).unwrap_or(&1.0)
    }

    /// Get the value scale for a layer
    pub fn v_scale(&self, layer_idx: u32) -> f32 {
        *self.v_scales.get(&layer_idx).unwrap_or(&1.0)
    }

    /// Check if calibration has been performed
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Calibrate scales for a single layer from observed KV tensors
    ///
    /// Computes optimal scales based on the maximum absolute values in the tensors.
    /// Uses MLX operations for GPU-accelerated computation.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `keys` - Key tensor (any shape)
    /// * `values` - Value tensor (any shape)
    ///
    /// # Safety
    /// The key and value pointers must be valid MLX array handles.
    #[cfg(target_os = "macos")]
    pub unsafe fn calibrate_layer(
        &mut self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
    ) -> Result<(f32, f32), String> {
        // Compute max absolute values using MLX ops
        // SAFETY: compute_max_abs is unsafe, caller guarantees valid pointers
        let k_max = unsafe { Self::compute_max_abs(keys)? };
        let v_max = unsafe { Self::compute_max_abs(values)? };

        // Compute scales: scale = FP8_MAX / max_abs
        // Higher max_abs -> lower scale (to fit in FP8 range)
        let k_scale = Self::compute_scale(k_max);
        let v_scale = Self::compute_scale(v_max);

        // Store scales
        self.k_scales.insert(layer_idx, k_scale);
        self.v_scales.insert(layer_idx, v_scale);
        self.calibrated = true;

        Ok((k_scale, v_scale))
    }

    /// Update scales using exponential moving average
    ///
    /// This is useful for online calibration during inference.
    /// The scale is updated as: new_scale = alpha * observed_scale + (1-alpha) * old_scale
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `keys` - Key tensor
    /// * `values` - Value tensor
    ///
    /// # Safety
    /// - `keys` and `values` must be valid MLX array pointers
    /// - Arrays must remain valid until this function returns
    #[cfg(target_os = "macos")]
    pub unsafe fn update_layer_ema(
        &mut self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
    ) -> Result<(f32, f32), String> {
        // Compute current max absolute values
        // SAFETY: compute_max_abs is unsafe, caller guarantees valid pointers
        let k_max = unsafe { Self::compute_max_abs(keys)? };
        let v_max = unsafe { Self::compute_max_abs(values)? };

        // Update running max using EMA
        let k_running = self.k_running_max.entry(layer_idx).or_insert(k_max);
        *k_running = self.alpha * k_max + (1.0 - self.alpha) * *k_running;

        let v_running = self.v_running_max.entry(layer_idx).or_insert(v_max);
        *v_running = self.alpha * v_max + (1.0 - self.alpha) * *v_running;

        // Compute scales from running max
        let k_scale = Self::compute_scale(*k_running);
        let v_scale = Self::compute_scale(*v_running);

        // Store scales
        self.k_scales.insert(layer_idx, k_scale);
        self.v_scales.insert(layer_idx, v_scale);
        self.calibrated = true;

        Ok((k_scale, v_scale))
    }

    /// Compute max absolute value of a tensor using MLX ops
    #[cfg(target_os = "macos")]
    unsafe fn compute_max_abs(tensor: *mut mlx_sys::mlx_array) -> Result<f32, String> {
        use mlx_sys::{
            mlx_array_abs, mlx_array_delete, mlx_array_eval, mlx_array_item_at_float32,
            mlx_array_max,
        };

        if tensor.is_null() {
            return Err("Null tensor pointer".to_string());
        }

        // Compute abs(tensor)
        let abs_tensor = unsafe { mlx_array_abs(tensor) };
        if abs_tensor.is_null() {
            return Err("Failed to compute abs".to_string());
        }

        // Compute max over all dimensions (pass null for axes = all axes)
        let max_tensor = unsafe { mlx_array_max(abs_tensor, std::ptr::null(), 0, false) };
        if max_tensor.is_null() {
            // Clean up abs_tensor before returning error
            unsafe { mlx_array_delete(abs_tensor) };
            return Err("Failed to compute max".to_string());
        }

        // Evaluate to get the result
        unsafe { mlx_array_eval(max_tensor) };

        // Extract scalar value (index 0 for scalar result)
        let mut max_val: f32 = 0.0;
        let ok = unsafe { mlx_array_item_at_float32(max_tensor, 0, &mut max_val) };

        // Clean up intermediate arrays
        unsafe {
            mlx_array_delete(abs_tensor);
            mlx_array_delete(max_tensor);
        }

        if !ok {
            return Err("Failed to extract max value".to_string());
        }

        Ok(max_val)
    }

    /// Compute scale from max absolute value
    fn compute_scale(max_abs: f32) -> f32 {
        if max_abs < MIN_SCALE {
            // Tensor is essentially zero, use default scale
            1.0
        } else {
            // scale = FP8_MAX / max_abs
            // This ensures max_abs * scale = FP8_MAX (fits exactly in FP8 range)
            FP8_E4M3_MAX / max_abs
        }
    }

    /// Set scales directly (useful for loading pre-calibrated values)
    pub fn set_scales(&mut self, layer_idx: u32, k_scale: f32, v_scale: f32) {
        self.k_scales.insert(layer_idx, k_scale);
        self.v_scales.insert(layer_idx, v_scale);
        self.calibrated = true;
    }

    /// Get all scales as vectors (for serialization)
    pub fn get_all_scales(&self) -> (Vec<f32>, Vec<f32>) {
        let mut k_scales = vec![1.0; self.num_layers as usize];
        let mut v_scales = vec![1.0; self.num_layers as usize];

        for (&layer_idx, &scale) in &self.k_scales {
            if (layer_idx as usize) < k_scales.len() {
                k_scales[layer_idx as usize] = scale;
            }
        }

        for (&layer_idx, &scale) in &self.v_scales {
            if (layer_idx as usize) < v_scales.len() {
                v_scales[layer_idx as usize] = scale;
            }
        }

        (k_scales, v_scales)
    }

    /// Load scales from vectors (for deserialization)
    pub fn load_scales(&mut self, k_scales: &[f32], v_scales: &[f32]) {
        for (layer_idx, &scale) in k_scales.iter().enumerate() {
            self.k_scales.insert(layer_idx as u32, scale);
        }

        for (layer_idx, &scale) in v_scales.iter().enumerate() {
            self.v_scales.insert(layer_idx as u32, scale);
        }

        self.calibrated = true;
    }

    /// Reset calibration state
    pub fn reset(&mut self) {
        self.k_scales.clear();
        self.v_scales.clear();
        self.k_running_max.clear();
        self.v_running_max.clear();
        self.calibrated = false;
    }

    /// Get statistics about the scales
    pub fn stats(&self) -> KvScaleStats {
        let k_scales: Vec<f32> = self.k_scales.values().copied().collect();
        let v_scales: Vec<f32> = self.v_scales.values().copied().collect();

        KvScaleStats {
            num_layers_calibrated: self.k_scales.len() as u32,
            k_scale_min: k_scales.iter().copied().fold(f32::INFINITY, f32::min),
            k_scale_max: k_scales.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            k_scale_mean: if k_scales.is_empty() {
                0.0
            } else {
                k_scales.iter().sum::<f32>() / k_scales.len() as f32
            },
            v_scale_min: v_scales.iter().copied().fold(f32::INFINITY, f32::min),
            v_scale_max: v_scales.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            v_scale_mean: if v_scales.is_empty() {
                0.0
            } else {
                v_scales.iter().sum::<f32>() / v_scales.len() as f32
            },
        }
    }
}

/// Statistics about KV scales
#[derive(Debug, Clone)]
pub struct KvScaleStats {
    /// Number of layers with calibrated scales
    pub num_layers_calibrated: u32,
    /// Minimum key scale
    pub k_scale_min: f32,
    /// Maximum key scale
    pub k_scale_max: f32,
    /// Mean key scale
    pub k_scale_mean: f32,
    /// Minimum value scale
    pub v_scale_min: f32,
    /// Maximum value scale
    pub v_scale_max: f32,
    /// Mean value scale
    pub v_scale_mean: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_manager_creation() {
        let manager = KvScaleManager::new(28);
        assert_eq!(manager.num_layers, 28);
        assert!(!manager.is_calibrated());
        assert_eq!(manager.k_scale(0), 1.0);
        assert_eq!(manager.v_scale(0), 1.0);
    }

    #[test]
    fn test_default_scales() {
        let mut manager = KvScaleManager::new(4);
        manager.init_default_scales();

        for i in 0..4 {
            assert_eq!(manager.k_scale(i), 1.0);
            assert_eq!(manager.v_scale(i), 1.0);
        }
    }

    #[test]
    fn test_set_scales() {
        let mut manager = KvScaleManager::new(4);
        manager.set_scales(0, 0.5, 0.25);
        manager.set_scales(1, 2.0, 1.5);

        assert_eq!(manager.k_scale(0), 0.5);
        assert_eq!(manager.v_scale(0), 0.25);
        assert_eq!(manager.k_scale(1), 2.0);
        assert_eq!(manager.v_scale(1), 1.5);
        assert!(manager.is_calibrated());
    }

    #[test]
    fn test_compute_scale() {
        // max_abs = 448 -> scale = 1.0
        assert!((KvScaleManager::compute_scale(448.0) - 1.0).abs() < 1e-6);

        // max_abs = 224 -> scale = 2.0
        assert!((KvScaleManager::compute_scale(224.0) - 2.0).abs() < 1e-6);

        // max_abs = 896 -> scale = 0.5
        assert!((KvScaleManager::compute_scale(896.0) - 0.5).abs() < 1e-6);

        // Very small max_abs -> scale = 1.0 (default)
        assert_eq!(KvScaleManager::compute_scale(0.0), 1.0);
    }

    #[test]
    fn test_serialization() {
        let mut manager = KvScaleManager::new(4);
        manager.set_scales(0, 0.5, 0.25);
        manager.set_scales(1, 2.0, 1.5);
        manager.set_scales(2, 1.0, 1.0);
        manager.set_scales(3, 0.8, 0.9);

        let (k_scales, v_scales) = manager.get_all_scales();

        assert_eq!(k_scales, vec![0.5, 2.0, 1.0, 0.8]);
        assert_eq!(v_scales, vec![0.25, 1.5, 1.0, 0.9]);

        // Test loading
        let mut new_manager = KvScaleManager::new(4);
        new_manager.load_scales(&k_scales, &v_scales);

        for i in 0..4 {
            assert_eq!(new_manager.k_scale(i), manager.k_scale(i));
            assert_eq!(new_manager.v_scale(i), manager.v_scale(i));
        }
    }

    #[test]
    fn test_stats() {
        let mut manager = KvScaleManager::new(4);
        manager.set_scales(0, 0.5, 0.25);
        manager.set_scales(1, 2.0, 1.5);
        manager.set_scales(2, 1.0, 1.0);
        manager.set_scales(3, 0.5, 0.25);

        let stats = manager.stats();
        assert_eq!(stats.num_layers_calibrated, 4);
        assert_eq!(stats.k_scale_min, 0.5);
        assert_eq!(stats.k_scale_max, 2.0);
        assert!((stats.k_scale_mean - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut manager = KvScaleManager::new(4);
        manager.set_scales(0, 0.5, 0.25);
        assert!(manager.is_calibrated());

        manager.reset();
        assert!(!manager.is_calibrated());
        assert_eq!(manager.k_scale(0), 1.0);
    }

    #[test]
    fn test_with_alpha() {
        let manager = KvScaleManager::with_alpha(4, 0.5);
        assert_eq!(manager.alpha, 0.5);

        // Test clamping
        let manager = KvScaleManager::with_alpha(4, -0.5);
        assert_eq!(manager.alpha, 0.0);

        let manager = KvScaleManager::with_alpha(4, 1.5);
        assert_eq!(manager.alpha, 1.0);
    }
}
