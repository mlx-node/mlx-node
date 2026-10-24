//! Opt-in staged K-walk spreading on the exact target projection shapes.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use std::ffi::CString;

fn tiled_call(x: &MxArray, w: &Weights, q: &KQuant) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(format!("{}@t64", q.mode)).unwrap();
    unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            q.group_size,
            q.bits,
            mode.as_ptr(),
            GPU,
        )
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "isolated GPU microbenchmark; run separate off/on processes, no concurrent model inference"]
fn actual_target_projection_microbenchmark() {
    assert!(gpu_gen() >= 17 && nax_available());
    assert_eq!(
        std::env::var("MLX_KQUANT_REGISTER_OUTPUT").as_deref(),
        Ok("0"),
        "isolate staged spreading from register MMA"
    );
    let mut cases = vec![
        ("gdn_qkv", "q6k", 8192, 2048),
        ("gdn_z", "q4k", 4096, 2048),
        ("attention_v", "q6k", 512, 2048),
        ("mixer_output", "q4k", 2048, 4096),
    ];
    if std::env::var("MLX_STAGED_SPREAD_HEAD").as_deref() == Ok("1") {
        cases.push(("vocabulary_head", "q6k", 248320, 2048));
    }
    for (label, mode, n, k) in cases {
        let q = KQUANTS.iter().find(|q| q.mode == mode).unwrap();
        let mut original = weights(q, &[n], k, 0x9411);
        original.biases = original
            .biases
            .mul_scalar(0.001)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let w = Weights {
            w: kquant_tile_rows(&original.w, i64::from(q.bits)).unwrap(),
            scales: kquant_tile_rows(&original.scales, 16).unwrap(),
            biases: kquant_tile_rows(&original.biases, q.biases_cols).unwrap(),
        };
        let x = activation(&[8, k], 0x9472, DType::BFloat16);
        start_counting();
        read_output("staged spread dispatch proof", tiled_call(&x, &w, q));
        assert_eq!(
            family_count("qmm_m8_nax_t64"),
            1,
            "{label} must dispatch through qmm_m8_nax_t64"
        );
        stop_counting();
        let batch = if label == "vocabulary_head" { 8 } else { 64 };
        let evaluate = || {
            let mut outputs: Vec<_> = (0..batch).map(|_| tiled_call(&x, &w, q)).collect();
            assert!(outputs.iter().all(|h| !h.is_null()));
            let mut error = [0i8; 512];
            let start = std::time::Instant::now();
            assert!(unsafe {
                mlx_sys::mlx_eval_with_error(
                    outputs.as_mut_ptr(),
                    outputs.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            });
            let microseconds = start.elapsed().as_secs_f64() * 1e6 / batch as f64;
            let last = outputs.pop().unwrap();
            for output in outputs {
                unsafe { mlx_sys::mlx_array_delete(output) };
            }
            let (_, _, bits) = read_output("projection", last);
            (microseconds, bits)
        };
        for _ in 0..10 {
            evaluate();
        }
        let mut times = Vec::with_capacity(100);
        let mut hash = 0xcbf29ce484222325u64;
        for _ in 0..100 {
            let (time, bits) = evaluate();
            times.push(time);
            for value in bits {
                hash = (hash ^ u64::from(value)).wrapping_mul(0x100000001b3);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "spread={} name={label} mode={mode} M=8 N={n} K={k} median_us={:.3} p10_us={:.3} hash={hash:x}",
            std::env::var("MLX_KQUANT_STAGED_SPREAD").unwrap_or_else(|_| "0".into()),
            times[50],
            times[10]
        );
    }
}
