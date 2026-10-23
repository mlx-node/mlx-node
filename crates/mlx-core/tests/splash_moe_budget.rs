//! Device row budgets preserve physical verifier shape across compiled replays.
mod kquant_support;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use std::sync::atomic::{AtomicUsize, Ordering};

const H: i64 = 1024;
const I: i64 = 512;
const E: i64 = 8;
const M: i64 = 8;
const TOP: i32 = 2;

fn tile(a: &MxArray, unit: i64) -> MxArray {
    let shape = a.shape().unwrap();
    kquant_tile_rows(&a.reshape(&[-1, *shape.last().unwrap()]).unwrap(), unit)
        .unwrap()
        .reshape(&shape)
        .unwrap()
}

struct Builder {
    formats: [u32; 6],
    traces: AtomicUsize,
}

unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || n_inputs != 22 || n_outputs != 1 {
        return false;
    }
    let context = unsafe { &*context.cast::<Builder>() };
    let handles: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n_inputs) }
        .iter()
        .map(|&h| h.cast_mut())
        .collect();
    context.traces.fetch_add(1, Ordering::Relaxed);
    let result = unsafe {
        mlx_sys::mlx_splash_moe(
            handles.as_ptr(),
            handles.len(),
            context.formats.as_ptr(),
            TOP,
            true,
        )
    };
    for handle in handles {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    unsafe { *outputs = result };
    !result.is_null()
}

struct Compiled(u64);
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}

fn call(arrays: &[*mut mlx_sys::mlx_array], formats: &[u32; 6]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_splash_moe(arrays.as_ptr(), arrays.len(), formats.as_ptr(), TOP, true) }
}

#[cfg(target_os = "macos")]
#[test]
fn budget_replay_masks_suffix_and_crosses_expert_occupancy_tiers() {
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        eprintln!("grouped MoE unavailable; requires grouped Metal kernels");
        return;
    }
    let modes = ["q4k", "q5k", "q6k", "q6k", "q4k", "q5k"];
    let packed: Vec<_> = modes
        .iter()
        .enumerate()
        .map(|(j, mode)| {
            let kq = KQUANTS.iter().find(|k| k.mode == *mode).unwrap();
            let (n, k) = if j % 3 == 2 { (H, I) } else { (I, H) };
            let leading = if j < 3 { vec![E, n] } else { vec![n] };
            let mut w = weights(kq, &leading, k, 811 + j as u32);
            w.biases = w
                .biases
                .mul_scalar(0.001)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            w
        })
        .collect();
    let x = activation(&[M, H], 829, DType::BFloat16);
    // All rows choose the same two experts. R=1 exercises the sparse column
    // kernel, while R=3 and R=8 exercise the staged live-tile path.
    let logits: Vec<_> = (0..M)
        .flat_map(|_| (0..E).map(|e| 2.0 - e as f32 * 0.7))
        .collect();
    let shared: Vec<_> = (0..M).map(|r| r as f32 * 0.1 - 0.3).collect();
    for tiled in [false, true] {
        let planes: Vec<_> = packed
            .iter()
            .zip(modes)
            .flat_map(|(w, mode)| {
                let kq = KQUANTS.iter().find(|k| k.mode == mode).unwrap();
                if tiled {
                    vec![
                        tile(&w.w, i64::from(kq.bits)),
                        tile(&w.scales, 16),
                        tile(&w.biases, kq.biases_cols),
                    ]
                } else {
                    vec![w.w.clone(), w.scales.clone(), w.biases.clone()]
                }
            })
            .collect();
        for (dtype_index, dtype) in [DType::BFloat16, DType::Float32].into_iter().enumerate() {
            let logits = MxArray::from_float32(&logits, &[M, E])
                .unwrap()
                .astype(dtype)
                .unwrap();
            let shared = MxArray::from_float32(&shared, &[M, 1])
                .unwrap()
                .astype(dtype)
                .unwrap();
            let mut context = Builder {
                formats: modes.map(|mode| {
                    KQUANTS.iter().find(|k| k.mode == mode).unwrap().bits as u32
                        | if tiled { 256 } else { 0 }
                }),
                traces: AtomicUsize::new(0),
            };
            let mut arrays = vec![x.as_raw_ptr(), logits.as_raw_ptr(), shared.as_raw_ptr()];
            arrays.extend(planes.iter().map(MxArray::as_raw_ptr));
            let baseline = read_output("all rows", call(&arrays, &context.formats));
            let values = |bits: &[u32]| -> Vec<f32> {
                bits.iter()
                    .map(|&v| {
                        if dtype == DType::Float32 {
                            f32::from_bits(v)
                        } else {
                            half::bf16::from_bits(v as u16).to_f32()
                        }
                    })
                    .collect()
            };
            let baseline_values = values(&baseline.2);
            let compiled =
                Compiled(0xB0D6_3500_0000_0000 | (u64::from(tiled) << 8) | dtype_index as u64);
            for rows in [1, 3, 8, 1, 8, 3] {
                let budget = MxArray::from_int32(&[rows], &[1]).unwrap();
                arrays.push(budget.as_raw_ptr());
                let eager = read_output("budget eager", call(&arrays, &context.formats));
                let inputs: Vec<_> = arrays.iter().map(|&h| h.cast_const()).collect();
                let mut output = std::ptr::null_mut();
                assert!(unsafe {
                    mlx_sys::mlx_compiled_graph_invoke(
                        compiled.0,
                        Some(build),
                        (&mut context as *mut Builder).cast(),
                        inputs.as_ptr(),
                        inputs.len(),
                        &mut output,
                        1,
                        true,
                    )
                });
                let replay = read_output("budget replay", output);
                assert_eq!(replay, eager, "compiled replay changed budget R={rows}");
                assert_eq!(
                    context.traces.load(Ordering::Relaxed),
                    1,
                    "budget contents must not retrace"
                );
                assert_eq!(eager.0, [M, H]);
                assert_eq!(eager.1, dtype);
                let live = rows as usize * H as usize;
                assert!(
                    eager.2[live..].iter().all(|&b| b == 0),
                    "dead rows must be initialized positive zero"
                );
                let actual = values(&eager.2);
                assert!(actual.iter().all(|x| x.is_finite()));
                if rows == M as i32 {
                    assert_eq!(
                        eager.2, baseline.2,
                        "R=M must preserve the original path exactly"
                    );
                } else {
                    // Changing live expert occupancy can change the reduction
                    // order, but must retain the same BF16 projection contract.
                    let peak = baseline_values[..live]
                        .iter()
                        .fold(0.0f32, |a, b| a.max(b.abs()))
                        .max(1e-8);
                    let error = actual[..live]
                        .iter()
                        .zip(&baseline_values[..live])
                        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
                    let (squared_error, squared_reference) = actual[..live]
                        .iter()
                        .zip(&baseline_values[..live])
                        .fold((0.0f64, 0.0f64), |(e, r), (&x, &y)| {
                            let difference = f64::from(x) - f64::from(y);
                            (e + difference * difference, r + f64::from(y).powi(2))
                        });
                    let relative_rms =
                        (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
                    println!(
                        "budget R={rows} tiled={tiled} dtype={dtype:?} max_error={error} peak={peak} normalized_peak_error={} relative_rms={relative_rms}",
                        error / peak
                    );
                    assert!(
                        error <= 0.01 * peak,
                        "R={rows} prefix changed: error={error} peak={peak} tiled={tiled} dtype={dtype:?}"
                    );
                }
                arrays.pop();
            }
        }
    }
}
