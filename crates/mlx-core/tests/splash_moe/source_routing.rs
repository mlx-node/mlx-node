use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
struct Builder {
    modes: Vec<u32>,
    residual: bool,
    traces: AtomicUsize,
}
unsafe extern "C" fn build(
    ctx: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if ctx.is_null() || m != 1 {
        return false;
    }
    let b = unsafe { &*ctx.cast::<Builder>() };
    if n != 22 + usize::from(b.residual) {
        return false;
    }
    b.traces.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let out = unsafe {
        mlx_sys::mlx_splash_moe_with_router(
            a.as_ptr(),
            22,
            b.modes.as_ptr(),
            8,
            true,
            if b.residual {
                a[22]
            } else {
                std::ptr::null_mut()
            },
        )
    };
    for h in a {
        unsafe { mlx_sys::mlx_array_delete(h) }
    }
    unsafe { *outputs = out };
    !out.is_null()
}
fn eager(a: &[MxArray], modes: &[u32], residual: bool) -> *mut mlx_sys::mlx_array {
    let handles: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
    unsafe {
        mlx_sys::mlx_splash_moe_with_router(
            handles.as_ptr(),
            22,
            modes.as_ptr(),
            8,
            true,
            if residual {
                handles[22]
            } else {
                std::ptr::null_mut()
            },
        )
    }
}
fn oracle(d: &std::path::Path, name: &str, got: &[u32], live: usize) {
    let bytes = std::fs::read(d.join(name)).unwrap();
    assert!(bytes.len() >= live * 2048 * 2);
    let expected: Vec<_> = bytes[..live * 2048 * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u32::from(u16::from_le_bytes(*b)))
        .collect();
    assert_eq!(
        &got[..live * 2048],
        expected.as_slice(),
        "original source live prefix {name}"
    );
    assert!(
        got[live * 2048..].iter().all(|b| *b == 0),
        "inactive source output must be exact zero"
    );
}
fn u32file(p: &std::path::Path) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}
pub(super) fn actual(saved: &[MxArray], modes: &[u32], layer: usize) {
    assert_eq!(saved.len(), 21);
    let root =
        std::path::PathBuf::from(std::env::var("SPLASH_MOE_SOURCE_ROUTING_ORACLE_DIR").unwrap());
    let mut probe = unsafe {
        mlx_sys::mlx_test_splash_moe_source_routing(
            saved[0].as_raw_ptr(),
            saved[1].as_raw_ptr(),
            saved[2].as_raw_ptr(),
        )
    };
    assert!(!probe.is_null());
    assert!(unsafe { mlx_sys::mlx_eval(&mut probe, 1) });
    let mut got = vec![0; 2192];
    assert!(unsafe { mlx_sys::mlx_array_to_uint32(probe, got.as_mut_ptr(), got.len()) });
    unsafe { mlx_sys::mlx_array_delete(probe) };
    let d = root.join("budget8");
    let mut expected = u32file(&d.join("scores.f32"));
    expected.extend(u32file(&d.join("selected.u32")));
    expected.extend(u32file(&d.join("coefficients.f32")));
    assert_eq!(got, expected, "full graph router exact source layer{layer}");
    // The optional budget must not change the unbudgeted ABI or output.
    for residual in [false, true] {
        let handles: Vec<_> = saved.iter().map(MxArray::as_raw_ptr).collect();
        let out = unsafe {
            mlx_sys::mlx_splash_moe_with_router(
                handles.as_ptr(),
                21,
                modes.as_ptr(),
                8,
                true,
                if residual {
                    saved[0].as_raw_ptr()
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        let unbudgeted = read_output("source routing no budget", out);
        oracle(
            &d,
            if residual {
                "residual-output.bf16"
            } else {
                "output.bf16"
            },
            &unbudgeted.2,
            8,
        );
    }
    for residual in [false, true] {
        let key = 0x737263726f757400 + u64::from(residual);
        let mut b = Builder {
            modes: modes.to_vec(),
            residual,
            traces: AtomicUsize::new(0),
        };
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        for (index, live) in [8, 1, 3, 8].into_iter().enumerate() {
            let mut a = saved.to_vec();
            a.push(MxArray::from_int32(&[live], &[]).unwrap());
            if residual {
                a.push(saved[0].clone())
            }
            let dir = root.join(format!("budget{live}"));
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
            assert_eq!(meta["layer"], layer);
            assert_eq!(meta["routing_mode"], "original");
            assert_eq!(
                meta["snapshot"].as_str(),
                Some(std::env::var("SPLASH_MOE_INPUT").unwrap().as_str())
            );
            let eager = read_output("source routing eager", eager(&a, modes, residual));
            assert_eq!(eager.0, [8, 2048]);
            assert_eq!(eager.1, DType::BFloat16);
            oracle(
                &dir,
                if residual {
                    "residual-output.bf16"
                } else {
                    "output.bf16"
                },
                &eager.2,
                live as usize,
            );
            let handles: Vec<_> = a.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
            let mut out = std::ptr::null_mut();
            assert!(unsafe {
                mlx_sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build),
                    (&mut b as *mut Builder).cast(),
                    handles.as_ptr(),
                    handles.len(),
                    &mut out,
                    1,
                    true,
                )
            });
            start_counting();
            let replay = read_output("source routing replay", out);
            assert_eq!(family_count("splash_moe_source_routing"), 1);
            assert_eq!(family_count("splash_moe_source_math"), 1);
            stop_counting();
            assert_eq!(replay, eager);
            assert_eq!(b.traces.load(Ordering::Relaxed), 1);
            // Device budget is a graph input. Physical M and its precision tier stay fixed.
            eprintln!(
                "SOURCE_ROUTING_EXACT layer={layer} residual={residual} live={live} replay={index}"
            );
        }
        // Reuse the same graph with changed x and both weight matrices, including
        // a valid nonzero contiguous offset. Compare eager/compiled exact bits.
        let mut a = saved.to_vec();
        for (j, value) in a.iter_mut().enumerate().take(3) {
            let changed = value
                .mul_scalar(0.9375 + j as f64 * 0.03125)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let shape = changed.shape().unwrap();
            let flat = changed.reshape(&[-1]).unwrap();
            let pad = MxArray::zeros(&[1], Some(DType::BFloat16)).unwrap();
            let size = shape.iter().product::<i64>();
            let storage = MxArray::concatenate_many(vec![&pad, &flat], Some(0)).unwrap();
            storage.eval();
            *value = storage
                .slice(&[1], &[size + 1])
                .unwrap()
                .reshape(&shape)
                .unwrap();
            value.eval();
            assert_ne!(
                unsafe { mlx_sys::mlx_array_get_buffer_offset(value.as_raw_ptr()) } % 16,
                0
            );
        }
        a.push(MxArray::from_int32(&[3], &[]).unwrap());
        if residual {
            a.push(a[0].clone())
        }
        let expected = read_output("changed source routing eager", eager(&a, modes, residual));
        let h: Vec<_> = a.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut b as *mut Builder).cast(),
                h.as_ptr(),
                h.len(),
                &mut out,
                1,
                true,
            )
        });
        assert_eq!(
            read_output("changed source routing compiled", out),
            expected
        );
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        // Source-routed slots always mean weights. Invalid shapeless replay
        // is rejected, never reinterpreted as the native logits-input ABI.
        for m in [1, 7, 33] {
            let mut bad = a.clone();
            bad[0] = MxArray::zeros(&[m, 2048], Some(DType::BFloat16)).unwrap();
            if residual {
                bad[22] = bad[0].clone()
            }
            assert!(eager(&bad, modes, residual).is_null());
            let h: Vec<_> = bad.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
            let mut failed = std::ptr::null_mut();
            let ok = unsafe {
                mlx_sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build),
                    (&mut b as *mut Builder).cast(),
                    h.as_ptr(),
                    h.len(),
                    &mut failed,
                    1,
                    true,
                )
            };
            if ok {
                let mut error = [0i8; 1024];
                assert!(!unsafe {
                    mlx_sys::mlx_eval_with_error(&mut failed, 1, error.as_mut_ptr(), error.len())
                })
            }
            if !failed.is_null() {
                unsafe { mlx_sys::mlx_array_delete(failed) }
            }
        }
        let mut recovered = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut b as *mut Builder).cast(),
                h.as_ptr(),
                h.len(),
                &mut recovered,
                1,
                true,
            )
        });
        assert_eq!(
            read_output("source routing replay recovery", recovered),
            expected
        );
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        for j in 0..3 {
            let mut bad = a.clone();
            bad[j] = bad[j].astype(DType::Float32).unwrap();
            assert!(eager(&bad, modes, residual).is_null())
        }
        let mut unsupported = modes.to_vec();
        unsupported[0] = (unsupported[0] & 256) | 5;
        assert!(eager(&a, &unsupported, residual).is_null());
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}
#[test]
#[ignore = "exclusive GPU; actual Ornith one-layer weights and original source route/body/residual fixtures"]
fn source_routing_actual_layer_original_source_and_budget_replay() {
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_ROUTING_ORACLE_DIR").is_some());
    assert_ne!(
        std::env::var("MLX_SPLASH_MOE_SOURCE_MATH").as_deref(),
        Ok("1"),
        "new explicit source router policy must work independently of process source-math flag"
    );
    grouped_actual_layer_stage_comparison();
}
