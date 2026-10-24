//! The production source router uses a dedicated Fast-math JIT library.
//! Raw source fixtures were emitted by the original Splash production kernels.
use mlx_core::array::{DType, MxArray};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
const WORDS: usize = 8 * 256 + 2 * 8 * 9;
fn load_bf(path: &Path, shape: &[i64]) -> MxArray {
    let bytes = std::fs::read(path).unwrap();
    let values: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
        .collect();
    assert_eq!(values.len(), shape.iter().product::<i64>() as usize);
    MxArray::from_float32(&values, shape)
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap()
}
fn words(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}
fn raw(a: &[MxArray]) -> *mut mlx_sys::mlx_array {
    unsafe {
        mlx_sys::mlx_test_splash_moe_source_routing(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
        )
    }
}
fn evaluate(mut p: *mut mlx_sys::mlx_array) -> Option<Vec<u32>> {
    if p.is_null() {
        return None;
    }
    let mut error = [0i8; 1024];
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut p, 1, error.as_mut_ptr(), error.len()) };
    let result = if ok {
        assert_eq!(unsafe { mlx_sys::mlx_array_size(p) }, WORDS);
        let mut out = vec![0; WORDS];
        assert!(unsafe { mlx_sys::mlx_array_to_uint32(p, out.as_mut_ptr(), out.len()) });
        Some(out)
    } else {
        None
    };
    unsafe { mlx_sys::mlx_array_delete(p) };
    result
}
fn view(a: &MxArray, mode: usize) -> MxArray {
    let shape = a.shape().unwrap();
    let values = a.astype(DType::Float32).unwrap().to_float32().unwrap();
    let (r, c) = (shape[0] as usize, shape[1] as usize);
    if mode == 1 {
        let mut padded = vec![0.0];
        padded.extend_from_slice(values.as_ref());
        let storage = MxArray::from_float32(&padded, &[padded.len() as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        storage.eval();
        let result = storage
            .slice(&[1], &[padded.len() as i64])
            .unwrap()
            .reshape(&shape)
            .unwrap();
        result.eval();
        assert_ne!(
            unsafe { mlx_sys::mlx_array_get_buffer_offset(result.as_raw_ptr()) } % 16,
            0
        );
        result
    } else if mode == 2 {
        let t: Vec<_> = (0..c)
            .flat_map(|j| {
                (0..r).map({
                    let values = &values;
                    move |i| values[i * c + j]
                })
            })
            .collect();
        MxArray::from_float32(&t, &[c as i64, r as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .transpose(Some(&[1, 0]))
            .unwrap()
    } else {
        a.clone()
    }
}
struct Builder {
    traces: AtomicUsize,
}
unsafe extern "C" fn build(
    ctx: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if ctx.is_null() || n != 3 || m != 1 {
        return false;
    }
    unsafe { &*ctx.cast::<Builder>() }
        .traces
        .fetch_add(1, Ordering::Relaxed);
    let a = unsafe { std::slice::from_raw_parts(inputs, n) };
    let out = unsafe {
        mlx_sys::mlx_test_splash_moe_source_routing(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
        )
    };
    for h in a {
        unsafe { mlx_sys::mlx_array_delete(h.cast_mut()) }
    }
    unsafe { *outputs = out };
    !out.is_null()
}
fn compiled(b: &mut Builder, a: &[MxArray], key: u64) -> Option<Vec<u32>> {
    let handles: Vec<_> = a.iter().map(|v| v.as_raw_ptr().cast_const()).collect();
    let mut out = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_compiled_graph_invoke(
            key,
            Some(build),
            (b as *mut Builder).cast(),
            handles.as_ptr(),
            handles.len(),
            &mut out,
            1,
            true,
        )
    };
    if !ok {
        if !out.is_null() {
            unsafe { mlx_sys::mlx_array_delete(out) }
        };
        return None;
    }
    evaluate(out)
}
#[test]
#[ignore = "exclusive GPU; set SPLASH_SOURCE_ROUTER_FIXTURES to original-source fixture root"]
fn source_router_exact_original_fixtures_and_one_trace_replay() {
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_ROUTER_FIXTURES").unwrap());
    let mut b = Builder {
        traces: AtomicUsize::new(0),
    };
    let key = 0x736f757263657231;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    for (index, layer) in [0, 5, 10, 39, 0].into_iter().enumerate() {
        let d = root.join(format!("layer{layer}"));
        let original = [
            load_bf(&d.join("x.bf16"), &[8, 2048]),
            load_bf(&d.join("router.bf16"), &[256, 2048]),
            load_bf(&d.join("shared.bf16"), &[1, 2048]),
        ];
        let mut expected = words(&d.join("source-scores.f32"));
        expected.extend(words(&d.join("source-selected.u32")));
        expected.extend(words(&d.join("source-coefficients.f32")));
        assert_eq!(expected.len(), WORDS);
        let a: Vec<_> = original.iter().map(|x| view(x, index % 3)).collect();
        assert_eq!(evaluate(raw(&a)).unwrap(), expected, "source layer{layer}");
        assert_eq!(
            compiled(&mut b, &a, key).unwrap(),
            expected,
            "source replay layer{layer}"
        );
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        // Independent decoded-weight F64 score oracle, no source kernel reuse.
        let x = original[0]
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let w = original[1]
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let refs: Vec<f64> = (0..8 * 256)
            .map(|i| {
                (0..2048)
                    .map(|k| f64::from(x[i / 256 * 2048 + k]) * f64::from(w[i % 256 * 2048 + k]))
                    .sum()
            })
            .collect();
        let peak = refs.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let max = refs
            .iter()
            .zip(&expected)
            .map(|(e, b)| (e - f64::from(f32::from_bits(*b))).abs())
            .fold(0.0, f64::max);
        assert!(
            max <= peak * 3e-6 + 1e-7,
            "source router F64 max{max} peak{peak}"
        );
        if index == 0 {
            for width in [1, 7, 33] {
                let mut invalid = a.clone();
                invalid[0] = MxArray::zeros(&[width, 2048], Some(DType::BFloat16)).unwrap();
                assert!(raw(&invalid).is_null());
                assert!(compiled(&mut b, &invalid, key).is_none());
            }
            assert_eq!(
                compiled(&mut b, &a, key).unwrap(),
                expected,
                "valid replay recovers after rejection"
            );
        }
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}
#[test]
#[ignore = "exclusive M5 GPU; source-only shape/device contract"]
fn source_router_ties_nonfinite_and_contract_guards() {
    assert!(unsafe { mlx_sys::mlx_splash_moe_available() });
    let mut a = vec![
        MxArray::zeros(&[8, 2048], Some(DType::BFloat16)).unwrap(),
        MxArray::zeros(&[256, 2048], Some(DType::BFloat16)).unwrap(),
        MxArray::zeros(&[1, 2048], Some(DType::BFloat16)).unwrap(),
    ];
    let result = evaluate(raw(&a)).unwrap();
    for row in 0..8 {
        assert_eq!(
            &result[2048 + row * 9..2048 + row * 9 + 8],
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(result[2048 + row * 9 + 8], 256);
        assert_eq!(
            &result[2120 + row * 9..2120 + row * 9 + 8],
            &[0.125f32.to_bits(); 8]
        );
        assert_eq!(result[2120 + row * 9 + 8], 0.5f32.to_bits());
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        a[0] = MxArray::from_float32(&vec![bad; 8 * 2048], &[8, 2048])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let result = evaluate(raw(&a)).unwrap();
        for row in 0..8 {
            for slot in 0..8 {
                assert!(result[2048 + row * 9 + slot] < 256)
            }
            assert_eq!(result[2048 + row * 9 + 8], 256)
        }
    }
    for j in 0..3 {
        let mut bad = a.clone();
        bad[j] = bad[j].astype(DType::Float32).unwrap();
        assert!(raw(&bad).is_null())
    }
}
