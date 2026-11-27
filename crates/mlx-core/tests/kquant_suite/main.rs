//! Consolidated integration test suite for K-quant, mixed-affine, GDN,
//! DFlash2 attention, mixed-projection, and Splash MoE kernels.
//!
//! Sharing a single binary avoids compiling and linking 16 separate test
//! executables on every PR. Tests run serially under `RUST_TEST_THREADS=1`
//! (`.cargo/config.toml`), and any test that switches MLX's process-wide
//! default device uses [`DefaultDeviceGuard`] so Metal remains selected for
//! subsequent tests in this binary.

#[path = "../kquant_support/mod.rs"]
mod kquant_support;

#[path = "../affine_mixed_support/mod.rs"]
mod affine_mixed_support;

/// Restores MLX's process-wide default device and stream on drop so tests that
/// switch to CPU never leak CPU default device to later Metal tests in this
/// suite.
pub(crate) struct DefaultDeviceGuard {
    device: i32,
}

impl DefaultDeviceGuard {
    #[must_use]
    pub(crate) fn save() -> Self {
        // SAFETY: nullary query of MLX's default device code (0 = CPU, 1 = GPU).
        let device = unsafe { mlx_sys::mlx_default_device() };
        Self { device }
    }
}

impl Drop for DefaultDeviceGuard {
    fn drop(&mut self) {
        // SAFETY: restores the saved MLX default device and its default stream.
        unsafe {
            mlx_sys::mlx_set_default_device(self.device);
            let stream = mlx_sys::mlx_default_stream(self.device);
            mlx_sys::mlx_set_default_stream(stream);
        }
    }
}

mod affine_mixed_shapeless_compile;
mod dflash2_attention;
mod gdn_prepared_sg8;
mod gdn_schedule;
mod kquant_affine_tiled;
mod kquant_ggml_parity;
mod kquant_m8_nax;
mod kquant_mode_guards;
mod kquant_sg8;
mod kquant_shapeless_compile;
mod kquant_tiled;
mod mixed_projection;
mod splash_moe;
mod splash_moe_budget;
mod symmetric_bias_parity;
mod tiled_gather_m1;
