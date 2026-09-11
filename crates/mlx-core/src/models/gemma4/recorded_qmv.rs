//! Projection-only bandwidth probe. Inputs must be captured from the recorded
//! agent replay; no generated/padded activations or fabricated weights.
#![cfg(target_os = "macos")]

use crate::array::{DType, MxArray};
use crate::utils::safetensors::load_safetensors_lazy;
use std::collections::HashMap;
use std::path::PathBuf;

struct Projection {
    x: MxArray,
    w: MxArray,
    s: MxArray,
    b: MxArray,
    sf: MxArray,
    bf: MxArray,
}

#[test]
#[ignore = "requires Q4_0 model and real-session projection captures"]
fn recorded_gemma4_qmv_bandwidth() {
    let root = PathBuf::from(std::env::var("GEMMA4_QMV_MODEL").expect("prepared model path"));
    let captures = PathBuf::from(std::env::var("GEMMA4_QMV_INPUTS").expect("real capture path"));
    let variant =
        std::env::var("GEMMA4_QMV_VARIANT").expect("baseline, hoisted, mixed, or symmetric");
    assert!(["baseline", "hoisted", "mixed", "symmetric"].contains(&variant.as_str()));
    let mut weights = HashMap::new();
    for entry in std::fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            weights.extend(load_safetensors_lazy(path).unwrap());
        }
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["quantization"]["symmetric_zero_point"], 8);
    let mut files: Vec<_> = std::fs::read_dir(captures)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let mut projections = Vec::new();
    for path in files {
        let prefix = path.file_stem().unwrap().to_str().unwrap();
        let suffix = format!("{prefix}.weight");
        let keys: Vec<_> = weights.keys().filter(|k| k.ends_with(&suffix)).collect();
        assert_eq!(keys.len(), 1, "unique text projection {prefix}");
        let key = keys[0];
        let scale_key = key.trim_end_matches(".weight").to_string() + ".scales";
        let w = weights[key].clone();
        let s = weights[&scale_key].clone();
        assert_eq!(w.dtype().unwrap(), DType::Uint32);
        assert_eq!(s.dtype().unwrap(), DType::Float16);
        let x = load_safetensors_lazy(path).unwrap().remove("x").unwrap();
        assert_eq!(x.dtype().unwrap(), DType::BFloat16);
        assert_eq!(
            *x.shape().unwrap().last().unwrap(),
            w.shape_at(1).unwrap() * 8
        );
        let b = s.mul_scalar(-8.0).unwrap();
        let sf = s.astype(DType::Float32).unwrap();
        let bf = b.astype(DType::Float32).unwrap();
        projections.push(Projection { x, w, s, b, sf, bf });
    }
    assert_eq!(
        projections.len(),
        328,
        "complete recorded Gemma Q4_0 projection set"
    );
    // All loading and the exact metadata widening precede measurement.
    for p in &projections {
        MxArray::eval_arrays(&[&p.x, &p.w, &p.s, &p.b, &p.sf, &p.bf]).unwrap();
    }
    let run = || {
        let outputs: Vec<_> = projections
            .iter()
            .map(|p| {
                let raw = unsafe {
                    if variant == "mixed" || variant == "symmetric" {
                        mlx_sys::mlx_affine_qmv_bf16(
                            p.x.as_raw_ptr(),
                            p.w.as_raw_ptr(),
                            p.s.as_raw_ptr(),
                            if variant == "symmetric" {
                                std::ptr::null_mut()
                            } else {
                                p.b.as_raw_ptr()
                            },
                        )
                    } else {
                        let (s, b) = if variant == "hoisted" {
                            (&p.sf, &p.bf)
                        } else {
                            (&p.s, &p.b)
                        };
                        mlx_sys::mlx_quantized_matmul(
                            p.x.as_raw_ptr(),
                            p.w.as_raw_ptr(),
                            s.as_raw_ptr(),
                            b.as_raw_ptr(),
                            true,
                            32,
                            4,
                            c"affine".as_ptr(),
                        )
                    }
                };
                MxArray::from_handle(raw, "recorded qmv replay")
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
            })
            .collect();
        MxArray::eval_arrays(&outputs.iter().collect::<Vec<_>>()).unwrap();
        outputs
    };
    for _ in 0..10 {
        let _ = run();
    }
    let mut elapsed_ms = Vec::new();
    let mut last = Vec::new();
    for _ in 0..20 {
        let start = std::time::Instant::now();
        last = run();
        elapsed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    // Check finiteness outside the timed region.
    assert!(
        last.iter()
            .all(|x| x.to_float32().unwrap().iter().all(|v| v.is_finite()))
    );
    let mut sum_squared_error = 0.0f64;
    let mut sum_squared_reference = 0.0f64;
    let mut max_error = 0.0f32;
    let mut different = 0usize;
    let mut elements = 0usize;
    for (p, actual) in projections.iter().zip(&last) {
        let raw = unsafe {
            mlx_sys::mlx_quantized_matmul(
                p.x.as_raw_ptr(),
                p.w.as_raw_ptr(),
                p.sf.as_raw_ptr(),
                p.bf.as_raw_ptr(),
                true,
                32,
                4,
                c"affine".as_ptr(),
            )
        };
        let expected = MxArray::from_handle(raw, "reference projection")
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .to_float32()
            .unwrap();
        for (&a, &b) in actual.to_float32().unwrap().iter().zip(expected.iter()) {
            let error = (a - b).abs();
            max_error = max_error.max(error);
            sum_squared_error += f64::from(error).powi(2);
            sum_squared_reference += f64::from(b).powi(2);
            different += usize::from(a != b);
            elements += 1;
            assert!(
                error <= b.abs() * 0.0078125 + 0.00001,
                "recorded projection error exceeds a BF16 ULP: {a} vs {b}"
            );
        }
    }
    let relative_l2_error = (sum_squared_error / sum_squared_reference.max(1e-30)).sqrt();
    assert!(relative_l2_error < 0.001);
    let weight_bytes: usize = projections.iter().map(|p| p.w.nbytes()).sum();
    let sidecar_bytes: usize = projections
        .iter()
        .map(|p| {
            if variant == "hoisted" {
                p.sf.nbytes() + p.bf.nbytes()
            } else if variant == "symmetric" {
                p.s.nbytes()
            } else {
                p.s.nbytes() + p.b.nbytes()
            }
        })
        .sum();
    println!(
        "RECORDED_QMV {}",
        serde_json::json!({
            "variant": variant, "projections": projections.len(), "weightBytes": weight_bytes,
            "sidecarBytes": sidecar_bytes, "elapsedMs": elapsed_ms,
        "maxAbsoluteError": max_error, "relativeL2Error": relative_l2_error,
        "differentElements": different, "elements": elements,
            "scope": "Independent projection replay from actual agent activations; excludes attention, norms, tied Q6 head and token dependencies"
        })
    );
}
