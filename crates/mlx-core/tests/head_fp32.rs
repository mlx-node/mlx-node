//! Opt-in head precision experiment. Independent F64 decoded-weight oracle,
//! strict half-staged arithmetic oracle, and exact native BF16 cast boundary.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use mlx_core::models::quantized_linear::QuantizedLinear;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn call(x: &MxArray, q: &QuantizedLinear) -> (i32, *mut mlx_sys::mlx_array) {
    let mut out = std::ptr::null_mut();
    let status = unsafe {
        mlx_sys::mlx_kquant_head_f32(
            x.as_raw_ptr(),
            q.get_weight().as_raw_ptr(),
            q.get_scales().as_raw_ptr(),
            q.get_biases().unwrap().as_raw_ptr(),
            &mut out,
        )
    };
    (status, out)
}

#[test]
#[ignore = "head precision prototype; exclusive GPU and independent F64 oracle"]
fn head_f32_preserves_multiply_and_matches_cpu_oracle() {
    let _guard = SERIAL.lock().unwrap();
    assert!(nax_available());
    unsafe { std::env::set_var("MLX_DFLASH2_HEAD_FP32", "1") };
    let format = kquant("q6k");
    // Split and unsplit schedules, multiple K superblocks. The wide shape
    // reaches the same single-split tier as the actual 248320-column head.
    for (n, k) in [(512i64, 1024i64), (16384, 2048)] {
        let mut packed = weights(format, &[n], k, 0x8613);
        packed.biases = packed
            .biases
            .mul_scalar(0.001)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let q = QuantizedLinear::new(
            kquant_tile_rows(&packed.w, 6).unwrap(),
            kquant_tile_rows(&packed.scales, 16).unwrap(),
            Some(kquant_tile_rows(&packed.biases, 1).unwrap()),
            None,
            16,
            6,
            "q6k@t64".into(),
        );
        let x = activation(&[1, 8, k], 0x9891, DType::BFloat16);
        let xv = x.astype(DType::Float32).unwrap().to_float32().unwrap();
        let (status, handle) = call(&x, &q);
        assert_eq!(status, 1);
        let (shape, dtype, bits) = read_bits("FP32 head", handle);
        assert_eq!(shape, vec![1, 8, n]);
        assert_eq!(dtype, DType::Float32);
        let actual: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
        assert!(actual.iter().all(|v| v.is_finite()));
        let native = q
            .forward(&x)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let rounded: Vec<_> = actual
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        assert_eq!(
            rounded,
            native.to_vec(),
            "only terminal BF16 boundary may differ"
        );
        assert!(
            actual.iter().any(|v| v.to_bits() & 65535 != 0),
            "FP32 result was silently widened BF16"
        );
        let (_, _, decoded_bits) = read_bits(
            "CPU Q6 dequant",
            dequantize(&packed, DType::Float32, format, CPU),
        );
        let columns = [0usize, 1, 63, 64, n as usize / 2, n as usize - 1];
        let mut peak = 0.0f64;
        let mut stage_max = 0.0f64;
        let mut raw_max = 0.0f64;
        let mut raw_squares = 0.0f64;
        let mut ref_squares = 0.0f64;
        for row in 0..8usize {
            for &col in &columns {
                let mut raw = 0.0f64;
                let mut stage = 0.0f64;
                for j in 0..k as usize {
                    let w = f32::from_bits(decoded_bits[col * k as usize + j]);
                    let a = f64::from(xv[row * k as usize + j]);
                    raw += a * f64::from(w);
                    stage += a * f64::from(half::f16::from_f32(w).to_f32());
                }
                let got = f64::from(actual[row * n as usize + col]);
                peak = peak.max(raw.abs());
                stage_max = stage_max.max((got - stage).abs());
                raw_max = raw_max.max((got - raw).abs());
                raw_squares += (got - raw).powi(2);
                ref_squares += raw.powi(2);
            }
        }
        let rms = (raw_squares / ref_squares.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!(
            "head_f32 N={n} K={k} half_stage_peak={} exact_weight_peak={} exact_weight_rms={rms}",
            stage_max / peak,
            raw_max / peak
        );
        // Strict FP32 accumulation check against the *same* half-staged
        // weight values. Separately retain established raw decoded-weight
        // oracle bounds rather than treating native BF16 output as truth.
        assert!(stage_max <= peak * 0.00002 + 1e-7);
        assert!(raw_max <= peak * 0.01);
        assert!(rms <= 0.005);
        for rows in [1, 7, 9] {
            let tail = activation(&[1, rows, k], 711, DType::BFloat16);
            let (status, out) = call(&tail, &q);
            assert_eq!(status, 0);
            assert!(out.is_null());
        }
        let f32x = x.astype(DType::Float32).unwrap();
        let (status, out) = call(&f32x, &q);
        assert_eq!(status, 0);
        assert!(out.is_null());
        let previous = unsafe { mlx_sys::mlx_default_device() };
        unsafe { mlx_sys::mlx_set_default_device(0) };
        let (status, out) = call(&x, &q);
        unsafe { mlx_sys::mlx_set_default_device(previous) };
        assert_eq!(status, 0);
        assert!(out.is_null());
    }
    unsafe { std::env::remove_var("MLX_DFLASH2_HEAD_FP32") };
}

unsafe extern "C" fn build_head(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || n_inputs != 4 || n_outputs != 1 {
        return false;
    }
    unsafe { &*context.cast::<std::sync::atomic::AtomicUsize>() }
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n_inputs) }
        .iter()
        .map(|p| p.cast_mut())
        .collect();
    let status = unsafe { mlx_sys::mlx_kquant_head_f32(a[0], a[1], a[2], a[3], outputs) };
    for handle in a {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    status == 1
}
struct CompiledHead(u64);
impl Drop for CompiledHead {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}
#[test]
#[ignore = "head precision prototype; lazy compiled replay and alignment contract"]
fn head_f32_compiled_replay_tracks_data_and_misaligned_views() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let _guard = SERIAL.lock().unwrap();
    assert!(nax_available());
    unsafe { std::env::set_var("MLX_DFLASH2_HEAD_FP32", "1") };
    let key = CompiledHead(0xDFA2_F320_0000_0001);
    let mut traces = AtomicUsize::new(0);
    let format = kquant("q6k");
    let mut previous = None;
    for (iteration, seed) in [701u32, 727, 751].into_iter().enumerate() {
        let mut packed = weights(format, &[512], 1024, seed);
        packed.biases = packed
            .biases
            .mul_scalar(0.001)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let mut arrays = vec![
            activation(&[1, 8, 1024], seed + 19, DType::BFloat16),
            kquant_tile_rows(&packed.w, 6).unwrap(),
            kquant_tile_rows(&packed.scales, 16).unwrap(),
            kquant_tile_rows(&packed.biases, 1).unwrap(),
        ];
        let aligned: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
        let mut expected = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                mlx_sys::mlx_kquant_head_f32(
                    aligned[0],
                    aligned[1],
                    aligned[2],
                    aligned[3],
                    &mut expected,
                )
            },
            1
        );
        let expected = read_bits("aligned head", expected);
        if iteration == 1 {
            arrays = arrays
                .iter()
                .map(|input| {
                    let prefix = MxArray::zeros(&[1], Some(input.dtype().unwrap())).unwrap();
                    let storage =
                        MxArray::concatenate(&prefix, &input.reshape(&[-1]).unwrap(), 0).unwrap();
                    storage.eval();
                    let view = storage
                        .slice(&[1], &[input.size().unwrap() as i64 + 1])
                        .unwrap()
                        .reshape(&input.shape().unwrap())
                        .unwrap();
                    assert_ne!(
                        unsafe { mlx_sys::mlx_array_get_buffer_offset(view.as_raw_ptr()) } % 16,
                        0
                    );
                    view
                })
                .collect();
        }
        let inputs: Vec<_> = arrays.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key.0,
                Some(build_head),
                (&mut traces as *mut AtomicUsize).cast(),
                inputs.as_ptr(),
                inputs.len(),
                &mut out,
                1,
                true,
            )
        });
        let got = read_bits("compiled head", out);
        assert_eq!(
            got, expected,
            "changed head/input data or layout was not replayed"
        );
        if let Some(old) = previous {
            assert_ne!(got.2, old, "fixture must change output data");
        }
        previous = Some(got.2);
        assert_eq!(traces.load(Ordering::Relaxed), 1);
        // Factory decline is the Rust head wrapper's unchanged fallback signal.
        // Do not invoke an already-cached graph while toggling process flags.
        unsafe { std::env::set_var("MLX_DFLASH2_HEAD_FP32", "0") };
        let mut declined = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                mlx_sys::mlx_kquant_head_f32(
                    arrays[0].as_raw_ptr(),
                    arrays[1].as_raw_ptr(),
                    arrays[2].as_raw_ptr(),
                    arrays[3].as_raw_ptr(),
                    &mut declined,
                )
            },
            0
        );
        assert!(declined.is_null());
        unsafe { std::env::set_var("MLX_DFLASH2_HEAD_FP32", "1") };
    }
    unsafe { std::env::remove_var("MLX_DFLASH2_HEAD_FP32") };
}
