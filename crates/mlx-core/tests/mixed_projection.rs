//! The fused mixer dispatch must preserve the existing native projection
//! bits, including each segment's independent split-K reduction order.
mod kquant_support;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use mlx_core::models::quantized_linear::QuantizedLinear;
use std::sync::atomic::{AtomicUsize, Ordering};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn linear(mode: &str, n: i64, k: i64, seed: u32) -> QuantizedLinear {
    let format = kquant(mode);
    let w = weights(format, &[n], k, seed);
    QuantizedLinear::new(
        kquant_tile_rows(&w.w, i64::from(format.bits)).unwrap(),
        kquant_tile_rows(
            &w.scales,
            format.super_ratio() * format.scale_bytes_per_group(),
        )
        .unwrap(),
        Some(kquant_tile_rows(&w.biases, format.bias_entries_per_super_block()).unwrap()),
        None,
        format.group_size,
        format.bits,
        format!("{mode}@t64"),
    )
}

fn call(x: &MxArray, projections: &[QuantizedLinear]) -> (i32, Vec<*mut mlx_sys::mlx_array>) {
    let mut inputs = vec![x.as_raw_ptr()];
    let mut bits = Vec::new();
    for projection in projections {
        inputs.extend([
            projection.get_weight().as_raw_ptr(),
            projection.get_scales().as_raw_ptr(),
            projection.get_biases().unwrap().as_raw_ptr(),
        ]);
        bits.push(projection.bits() as u32);
    }
    let mut outputs = vec![std::ptr::null_mut(); projections.len()];
    let status = unsafe {
        mlx_sys::mlx_mixed_projection(
            inputs.as_ptr(),
            inputs.len(),
            bits.as_ptr(),
            projections.len() as u32,
            outputs.as_mut_ptr(),
        )
    };
    (status, outputs)
}

#[test]
fn mixed_target_projections_match_standalone_bits() {
    let _guard = SERIAL.lock().unwrap();
    if !nax_available() {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }
    // Exact 35B full-attention and GDN widths, plus small mixed-format grids
    // whose segments select different split counts from the wide segments.
    for (k, dimensions) in [
        (2048, vec![("q4k", 8192), ("q4k", 512), ("q6k", 512)]),
        (2048, vec![("q6k", 8192), ("q4k", 4096)]),
        (1024, vec![("q5k", 64), ("q6k", 192), ("q4k", 320)]),
    ] {
        let projections: Vec<_> = dimensions
            .iter()
            .enumerate()
            .map(|(i, (mode, n))| linear(mode, *n, k, 391 + i as u32))
            .collect();
        for seed in [17, 29, 43] {
            let x = activation(&[1, 8, k], seed, DType::BFloat16);
            let (status, outputs) = call(&x, &projections);
            assert_eq!(status, 1, "K={k} dimensions={dimensions:?}");
            for (i, output) in outputs.into_iter().enumerate() {
                let (shape, dtype, actual) = read_bits("mixed target projection", output);
                let reference = projections[i].forward(&x).unwrap();
                assert_eq!(shape, reference.shape().unwrap().to_vec());
                assert_eq!(dtype, DType::BFloat16);
                let expected: Vec<_> = reference
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits() >> 16)
                    .collect();
                assert_eq!(actual, expected, "K={k} segment={i} seed={seed}");
            }
        }
    }
}

#[test]
fn mixed_misaligned_contiguous_inputs_match_aligned() {
    let _guard = SERIAL.lock().unwrap();
    if !nax_available() {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }
    let projections = vec![
        linear("q4k", 64, 1024, 791),
        linear("q5k", 192, 1024, 809),
        linear("q6k", 320, 1024, 827),
    ];
    let mut aligned = vec![activation(&[1, 8, 1024], 739, DType::BFloat16)];
    for projection in &projections {
        aligned.extend([
            projection.get_weight().clone(),
            projection.get_scales().clone(),
            projection.get_biases().unwrap().clone(),
        ]);
    }
    let misaligned: Vec<_> = aligned
        .iter()
        .map(|input| {
            let prefix = MxArray::zeros(&[1], Some(input.dtype().unwrap())).unwrap();
            let flat = input.reshape(&[-1]).unwrap();
            let storage = MxArray::concatenate(&prefix, &flat, 0).unwrap();
            storage.eval();
            let view = storage
                .slice(&[1], &[input.size().unwrap() as i64 + 1])
                .unwrap()
                .reshape(&input.shape().unwrap())
                .unwrap();
            let offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(view.as_raw_ptr()) };
            assert_ne!(offset % 16, 0, "fixture must retain its misaligned offset");
            view
        })
        .collect();
    let invoke = |arrays: &[MxArray]| {
        let inputs: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
        let mut outputs = vec![std::ptr::null_mut(); 3];
        assert_eq!(
            unsafe {
                mlx_sys::mlx_mixed_projection(
                    inputs.as_ptr(),
                    inputs.len(),
                    [4u32, 5, 6].as_ptr(),
                    3,
                    outputs.as_mut_ptr(),
                )
            },
            1
        );
        outputs
            .into_iter()
            .map(|output| read_bits("mixed aligned-view parity", output))
            .collect::<Vec<_>>()
    };
    let expected = invoke(&aligned);
    // Each plane is independently offset, then all ten operands together.
    // The U8/I8 metadata offset also exercises a one-byte allocation slice.
    for index in 0..=aligned.len() {
        let mut inputs = aligned.clone();
        if index == aligned.len() {
            inputs.clone_from(&misaligned);
        } else {
            inputs[index] = misaligned[index].clone();
        }
        assert_eq!(invoke(&inputs), expected, "misaligned operand {index}");
    }
}

struct MixedTrace {
    bits: [u32; 2],
    builds: AtomicUsize,
}

unsafe extern "C" fn build_mixed(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    count: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || inputs.is_null() || outputs.is_null() || count != 7 || n_outputs != 2 {
        return false;
    }
    let trace = unsafe { &*context.cast::<MixedTrace>() };
    trace.builds.fetch_add(1, Ordering::Relaxed);
    let handles: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, count) }
        .iter()
        .map(|p| p.cast_mut())
        .collect();
    let status = unsafe {
        mlx_sys::mlx_mixed_projection(handles.as_ptr(), count, trace.bits.as_ptr(), 2, outputs)
    };
    for handle in handles {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    status == 1
}

struct MixedGraph(u64);
impl Drop for MixedGraph {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}

fn owned_operands(x: &MxArray, projections: &[QuantizedLinear]) -> Vec<MxArray> {
    let mut arrays = vec![x.clone()];
    for projection in projections {
        arrays.extend([
            projection.get_weight().clone(),
            projection.get_scales().clone(),
            projection.get_biases().unwrap().clone(),
        ]);
    }
    arrays
}

fn invoke_mixed(
    trace: &mut MixedTrace,
    key: u64,
    arrays: &[MxArray],
) -> (bool, [*mut mlx_sys::mlx_array; 2]) {
    let inputs: Vec<_> = arrays.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
    let mut outputs = [std::ptr::null_mut(); 2];
    let ok = unsafe {
        mlx_sys::mlx_compiled_graph_invoke(
            key,
            Some(build_mixed),
            (trace as *mut MixedTrace).cast(),
            inputs.as_ptr(),
            inputs.len(),
            outputs.as_mut_ptr(),
            2,
            true,
        )
    };
    (ok, outputs)
}

fn mixed_eval_failure(mut outputs: [*mut mlx_sys::mlx_array; 2]) -> String {
    let mut error = [0i8; 512];
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(outputs.as_mut_ptr(), 2, error.as_mut_ptr(), error.len())
    };
    for handle in outputs {
        if !handle.is_null() {
            unsafe { mlx_sys::mlx_array_delete(handle) };
        }
    }
    assert!(!ok, "mismatched replay reached a successful dispatch");
    let bytes: Vec<u8> = error
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn mixed_shapeless_replay_rejects_every_smaller_weight_plane() {
    let _guard = SERIAL.lock().unwrap();
    if !nax_available() {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }
    let x = activation(&[1, 8, 1024], 0x451, DType::BFloat16);
    let traced_projections = vec![
        linear("q4k", 2048, 1024, 0x471),
        linear("q6k", 512, 1024, 0x491),
    ];
    let smaller_projections = vec![
        linear("q4k", 1024, 1024, 0x4b1),
        linear("q6k", 256, 1024, 0x4d1),
    ];
    let traced = owned_operands(&x, &traced_projections);
    let smaller = owned_operands(&x, &smaller_projections);
    let key = 0x6D69_7865_6400_0001;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let _graph = MixedGraph(key);
    let mut trace = MixedTrace {
        bits: [4, 6],
        builds: AtomicUsize::new(0),
    };
    let (ok, outputs) = invoke_mixed(&mut trace, key, &traced);
    assert!(ok, "initial mixed projection trace failed");
    let expected: Vec<_> = outputs
        .into_iter()
        .enumerate()
        .map(|(i, output)| {
            let out = read_output("mixed replay trace", output);
            assert_eq!(
                out.0,
                [1, 8, traced_projections[i].get_weight().shape().unwrap()[0]]
            );
            out
        })
        .collect();
    assert_eq!(trace.builds.load(Ordering::Relaxed), 1);

    for segment in 0..2 {
        for plane in 0..3 {
            let index = 1 + 3 * segment + plane;
            let mut bad = traced.clone();
            bad[index] = smaller[index].clone();
            let (ok, outputs) = invoke_mixed(&mut trace, key, &bad);
            assert!(
                ok,
                "segment {segment} plane {plane} did not use the cached graph"
            );
            let error = mixed_eval_failure(outputs);
            assert!(
                error.contains("mixed projection replay changed its weight-plane contract"),
                "segment {segment} plane {plane}: {error}"
            );
            assert_eq!(trace.builds.load(Ordering::Relaxed), 1);
        }
    }

    let (ok, outputs) = invoke_mixed(&mut trace, key, &traced);
    assert!(ok, "valid replay after rejection failed at construction");
    for (output, expected) in outputs.into_iter().zip(expected) {
        assert_eq!(read_output("mixed replay recovery", output), expected);
    }
    assert_eq!(trace.builds.load(Ordering::Relaxed), 1);
}

#[test]
fn unsupported_width_dtype_and_cpu_decline_without_outputs() {
    let _guard = SERIAL.lock().unwrap();
    let projections = vec![linear("q4k", 64, 256, 31), linear("q6k", 64, 256, 47)];
    for (rows, dtype) in [
        (1, DType::BFloat16),
        (7, DType::BFloat16),
        (8, DType::Float32),
    ] {
        let x = activation(&[rows, 256], 19, dtype);
        let (status, outputs) = call(&x, &projections);
        assert_eq!(status, 0);
        assert!(outputs.iter().all(|p| p.is_null()));
    }
    let x = activation(&[8, 256], 23, DType::BFloat16);
    let previous = unsafe { mlx_sys::mlx_default_device() };
    unsafe { mlx_sys::mlx_set_default_device(0) };
    let (status, outputs) = call(&x, &projections);
    unsafe { mlx_sys::mlx_set_default_device(previous) };
    assert_eq!(status, 0);
    assert!(outputs.iter().all(|p| p.is_null()));
}

fn global_split_fixture(mode: &str, n: i64, k: i64, seed: u32) -> (Weights, QuantizedLinear) {
    let q = kquant(mode);
    let mut row = weights(q, &[n], k, seed);
    row.biases = row
        .biases
        .mul_scalar(0.001)
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
    let projection = QuantizedLinear::new(
        kquant_tile_rows(&row.w, i64::from(q.bits)).unwrap(),
        kquant_tile_rows(&row.scales, q.super_ratio() * q.scale_bytes_per_group()).unwrap(),
        Some(kquant_tile_rows(&row.biases, q.bias_entries_per_super_block()).unwrap()),
        None,
        q.group_size,
        q.bits,
        format!("{mode}@t64"),
    );
    (row, projection)
}

#[test]
#[ignore = "source-style global split experiment; bounded exact-dequantized F64 oracle"]
fn mixed_global_split_cpu_oracle() {
    let _guard = SERIAL.lock().unwrap();
    assert!(nax_available());
    for (k, dimensions) in [
        (2048, vec![("q4k", 8192), ("q4k", 512), ("q6k", 512)]),
        (2048, vec![("q6k", 8192), ("q4k", 4096)]),
        (4096, vec![("q5k", 4096), ("q6k", 4096)]),
    ] {
        let fixtures: Vec<_> = dimensions
            .iter()
            .enumerate()
            .map(|(i, (mode, n))| global_split_fixture(mode, *n, k, 0x9132 + i as u32))
            .collect();
        let (row_major, projections): (Vec<_>, Vec<_>) = fixtures.into_iter().unzip();
        let x = activation(&[1, 8, k], 0x7213, DType::BFloat16);
        let input = x.astype(DType::Float32).unwrap().to_float32().unwrap();
        // Sample output columns across both tile boundaries and each segment's
        // full width. The oracle sums exact decoded weights in F64, independent
        // of either GPU split/reduction schedule or half-staged intermediates.
        let mut reference = Vec::new();
        for ((mode, n), packed) in dimensions.iter().zip(&row_major) {
            let (_, _, decoded) = read_bits(
                "global split CPU weights",
                dequantize(packed, DType::Float32, kquant(mode), CPU),
            );
            let columns = [0usize, 1, 63, 64, *n as usize / 2, *n as usize - 1];
            let mut samples = Vec::new();
            for row in 0..8usize {
                for &column in &columns {
                    let sum = (0..k as usize)
                        .map(|j| {
                            f64::from(f32::from_bits(decoded[column * k as usize + j]))
                                * f64::from(input[row * k as usize + j])
                        })
                        .sum::<f64>();
                    samples.push((row * *n as usize + column, sum));
                }
            }
            reference.push(samples);
        }
        let mut baseline = Vec::new();
        for enabled in [false, true] {
            unsafe {
                std::env::set_var(
                    "MLX_MIXED_PROJECTION_GLOBAL_SPLIT",
                    if enabled { "1" } else { "0" },
                )
            };
            let (status, outputs) = call(&x, &projections);
            assert_eq!(status, 1);
            for (segment, output) in outputs.into_iter().enumerate() {
                let (_, dtype, bits) = read_output("global split GPU projection", output);
                assert_eq!(dtype, DType::BFloat16);
                let values: Vec<_> = bits
                    .iter()
                    .map(|v| f64::from(half::bf16::from_bits(*v as u16).to_f32()))
                    .collect();
                assert!(values.iter().all(|v| v.is_finite()));
                let mut peak = 0.0f64;
                let mut max_error = 0.0f64;
                let mut squared_error = 0.0f64;
                let mut squared_reference = 0.0f64;
                for &(index, want) in &reference[segment] {
                    let error = values[index] - want;
                    peak = peak.max(want.abs());
                    max_error = max_error.max(error.abs());
                    squared_error += error * error;
                    squared_reference += want * want;
                }
                let rms = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
                eprintln!(
                    "global_split={enabled} K={k} segment={segment} dimensions={dimensions:?} cpu_peak_rel={} cpu_rms={rms}",
                    max_error / peak
                );
                assert!(
                    max_error <= 0.01 * peak,
                    "exact-dequantized peak error exceeds 1%"
                );
                assert!(rms <= 0.005, "exact-dequantized RMS error exceeds 0.5%");
                if enabled {
                    let old: &Vec<f64> = &baseline[segment];
                    let delta = values
                        .iter()
                        .zip(old)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>();
                    let energy = old.iter().map(|v| v * v).sum::<f64>();
                    let changed = values.iter().zip(old).filter(|(a, b)| a != b).count();
                    eprintln!(
                        "global_split_delta K={k} segment={segment} changed={changed}/{} relative_rms={}",
                        values.len(),
                        (delta / energy).sqrt()
                    );
                } else {
                    baseline.push(values);
                }
            }
        }
    }
    unsafe { std::env::remove_var("MLX_MIXED_PROJECTION_GLOBAL_SPLIT") };
}

#[test]
#[ignore = "isolated batched timing; run separate processes with global split off/on"]
fn mixed_global_split_microbenchmark() {
    let _guard = SERIAL.lock().unwrap();
    assert!(nax_available());
    for (label, dimensions) in [
        ("FA", vec![("q4k", 8192), ("q4k", 512), ("q6k", 512)]),
        ("GDN", vec![("q6k", 8192), ("q4k", 4096)]),
    ] {
        let projections: Vec<_> = dimensions
            .iter()
            .enumerate()
            .map(|(i, (mode, n))| global_split_fixture(mode, *n, 2048, 0x9132 + i as u32).1)
            .collect();
        let x = activation(&[1, 8, 2048], 0x7213, DType::BFloat16);
        let mut samples = Vec::new();
        let mut checksum = 0u32;
        for iteration in 0..40 {
            let mut roots: Vec<_> = (0..32)
                .flat_map(|_| {
                    let (status, outputs) = call(&x, &projections);
                    assert_eq!(status, 1);
                    outputs
                })
                .collect();
            let started = std::time::Instant::now();
            assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
            let elapsed = started.elapsed().as_secs_f64() * 1e6 / 32.0;
            if iteration >= 10 {
                samples.push(elapsed);
            }
            let last = roots.pop().unwrap();
            for output in roots {
                unsafe { mlx_sys::mlx_array_delete(output) };
            }
            let (_, _, bits) = read_output("global split timed projection", last);
            checksum = checksum.wrapping_mul(16777619) ^ bits[0];
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "mixed_global_split={} family={label} median_us={} p10_us={} checksum={checksum}",
            std::env::var("MLX_MIXED_PROJECTION_GLOBAL_SPLIT").unwrap_or_else(|_| "0".into()),
            samples[15],
            samples[3]
        );
    }
}
