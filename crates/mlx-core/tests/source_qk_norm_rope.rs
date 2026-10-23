//! Original Splash production QK+RoPE bit fixtures; actual checkpoint F32
//! norm weights, deterministic projected inputs, explicit runtime positions.
use mlx_core::array::{DType, MxArray};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
fn u16s(path: &Path) -> Vec<u16> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect()
}
fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect()
}
fn bf_view(values: &[u16], heads: usize, view: usize) -> MxArray {
    let shape = [1, 8, heads as i64, 256];
    if view == 0 {
        return MxArray::from_bfloat16(values, &shape).unwrap();
    }
    // Independent holes in both row/head strides plus a nonzero byte offset.
    let (hp, dp) = (heads + 1, 258usize);
    let mut storage = vec![0x7fc1; 8 * hp * dp];
    for t in 0..8 {
        for h in 0..heads {
            for d in 0..256 {
                storage[(t * hp + h + 1) * dp + d + 1] = values[(t * heads + h) * 256 + d];
            }
        }
    }
    let base = MxArray::from_bfloat16(&storage, &[1, 8, hp as i64, dp as i64]).unwrap();
    base.eval();
    let a = base.slice(&[0, 0, 1, 1], &[1, 8, hp as i64, 257]).unwrap();
    a.eval();
    assert_ne!(
        unsafe { mlx_sys::mlx_array_get_buffer_offset(a.as_raw_ptr()) },
        0
    );
    a
}
fn float_view(values: &[f32], view: usize) -> MxArray {
    if view == 0 {
        return MxArray::from_float32(values, &[values.len() as i64]).unwrap();
    }
    let mut storage = vec![f32::NAN; values.len() * 2];
    for (i, &v) in values.iter().enumerate() {
        storage[2 * i + 1] = v;
    }
    let base = MxArray::from_float32(&storage, &[values.len() as i64, 2]).unwrap();
    base.eval();
    let a = base
        .slice(&[0, 1], &[values.len() as i64, 2])
        .unwrap()
        .reshape(&[values.len() as i64])
        .unwrap();
    a.eval();
    assert_eq!(
        unsafe { mlx_sys::mlx_array_get_buffer_offset(a.as_raw_ptr()) },
        4
    );
    a
}
fn fixture(d: &Path, view: usize) -> Vec<MxArray> {
    let packed = u16s(&d.join("packed.bf16"));
    assert_eq!(packed.len(), 8 * 9216);
    let mut q = vec![0; 8 * 16 * 256];
    let mut k = vec![0; 8 * 2 * 256];
    for t in 0..8 {
        for h in 0..16 {
            q[(t * 16 + h) * 256..(t * 16 + h + 1) * 256]
                .copy_from_slice(&packed[t * 9216 + h * 512..t * 9216 + h * 512 + 256]);
        }
        k[t * 512..(t + 1) * 512].copy_from_slice(&packed[t * 9216 + 8192..t * 9216 + 8704]);
    }
    let (q, k) = if view == 2 {
        // Production Q/gate interleaving, and K as a view of the same packed owner.
        let p = MxArray::from_bfloat16(&packed, &[1, 8, 9216]).unwrap();
        p.eval();
        let q = p
            .slice(&[0, 0, 0], &[1, 8, 8192])
            .unwrap()
            .reshape(&[1, 8, 16, 512])
            .unwrap()
            .slice(&[0, 0, 0, 0], &[1, 8, 16, 256])
            .unwrap();
        let k = p
            .slice(&[0, 0, 8192], &[1, 8, 8704])
            .unwrap()
            .reshape(&[1, 8, 2, 256])
            .unwrap();
        (q, k)
    } else {
        (bf_view(&q, 16, view), bf_view(&k, 2, view))
    };
    let pos = i32::from_le_bytes(
        std::fs::read(d.join("position.i32"))
            .unwrap()
            .try_into()
            .unwrap(),
    );
    let offsets = if view == 0 {
        MxArray::from_int32(&[pos], &[1]).unwrap()
    } else {
        let b = MxArray::from_int32(&[0, pos, 0], &[3]).unwrap();
        b.eval();
        b.slice(&[1], &[2]).unwrap()
    };
    vec![
        q,
        k,
        float_view(&f32s(&d.join("qnorm.f32")), view),
        float_view(&f32s(&d.join("knorm.f32")), view),
        float_view(&f32s(&d.join("frequency.f32")), view),
        offsets,
    ]
}
fn raw(a: &[MxArray]) -> [*mut mlx_sys::mlx_array; 2] {
    let mut out = [std::ptr::null_mut(); 2];
    let ok = unsafe {
        mlx_sys::mlx_source_qk_norm_rope(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
            a[3].as_raw_ptr(),
            a[4].as_raw_ptr(),
            a[5].as_raw_ptr(),
            out.as_mut_ptr(),
            out.as_mut_ptr().add(1),
        )
    };
    assert_eq!(ok, out.iter().all(|p| !p.is_null()));
    out
}
fn evaluate(mut out: [*mut mlx_sys::mlx_array; 2], inputs: &[MxArray]) -> Option<[Vec<u16>; 2]> {
    if out.iter().any(|p| p.is_null()) {
        for p in out {
            if !p.is_null() {
                unsafe { mlx_sys::mlx_array_delete(p) }
            }
        }
        return None;
    }
    let mut error = [0i8; 2048];
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(out.as_mut_ptr(), 2, error.as_mut_ptr(), error.len())
    };
    let result = if ok {
        let buffers = out.map(|p| unsafe { mlx_sys::mlx_array_get_metal_buffer(p) });
        assert!(!buffers[0].is_null() && !buffers[1].is_null());
        assert_ne!(buffers[0], buffers[1], "outputs share storage");
        for a in inputs {
            let b = unsafe { mlx_sys::mlx_array_get_metal_buffer(a.as_raw_ptr()) };
            assert!(!buffers.contains(&b), "output aliases input");
        }
        let mut q = vec![0u16; 8 * 16 * 256];
        let mut k = vec![0u16; 8 * 2 * 256];
        assert_eq!(unsafe { mlx_sys::mlx_array_size(out[0]) }, q.len());
        assert_eq!(unsafe { mlx_sys::mlx_array_size(out[1]) }, k.len());
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(out[0], q.as_mut_ptr(), q.len()) });
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(out[1], k.as_mut_ptr(), k.len()) });
        Some([q, k])
    } else {
        None
    };
    for p in out {
        unsafe { mlx_sys::mlx_array_delete(p) }
    }
    result
}
struct Builder(AtomicUsize);
struct OwnedInputs<'a>(&'a [*const mlx_sys::mlx_array]);
impl Drop for OwnedInputs<'_> {
    fn drop(&mut self) {
        for &p in self.0 {
            if !p.is_null() {
                unsafe { mlx_sys::mlx_array_delete(p.cast_mut()) }
            }
        }
    }
}
unsafe extern "C" fn build(
    ctx: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if inputs.is_null() && n != 0 {
        return false;
    }
    let a = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(inputs, n) }
    };
    let _owned = OwnedInputs(a);
    if ctx.is_null() || outputs.is_null() || n != 6 || m != 2 {
        return false;
    }
    unsafe { &*ctx.cast::<Builder>() }
        .0
        .fetch_add(1, Ordering::Relaxed);
    unsafe {
        mlx_sys::mlx_source_qk_norm_rope(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
            a[3].cast_mut(),
            a[4].cast_mut(),
            a[5].cast_mut(),
            outputs,
            outputs.add(1),
        )
    }
}
fn compiled(b: &mut Builder, a: &[MxArray], key: u64) -> Option<[Vec<u16>; 2]> {
    let inputs: Vec<_> = a.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
    let mut out = [std::ptr::null_mut(); 2];
    let ok = unsafe {
        mlx_sys::mlx_compiled_graph_invoke(
            key,
            Some(build),
            (b as *mut Builder).cast(),
            inputs.as_ptr(),
            6,
            out.as_mut_ptr(),
            2,
            true,
        )
    };
    if !ok {
        for p in out {
            if !p.is_null() {
                unsafe { mlx_sys::mlx_array_delete(p) }
            }
        }
        return None;
    }
    evaluate(out, a)
}
#[test]
#[ignore = "exclusive measured GPU; SPLASH_SOURCE_QK_FIXTURES original-source fixtures"]
fn source_qk_original_bits_and_one_trace_views_weights_positions() {
    assert!(unsafe { mlx_sys::mlx_source_qk_norm_rope_available() });
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_QK_FIXTURES").unwrap());
    let key = 0x73716b726f706531;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut b = Builder(AtomicUsize::new(0));
    let mut cases = 0;
    for layer in (3..40).step_by(4) {
        for pos in [2048u32, 0, 32768, u32::MAX] {
            for fv in [0, 1] {
                let d = root.join(format!("layer{layer}-m8-position{pos}-frequency{fv}"));
                let a = fixture(&d, cases % 3);
                let expected = [u16s(&d.join("q.bf16")), u16s(&d.join("k.bf16"))];
                assert_eq!(evaluate(raw(&a), &a).unwrap(), expected, "source {d:?}");
                assert_eq!(compiled(&mut b, &a, key).unwrap(), expected, "replay {d:?}");
                assert_eq!(b.0.load(Ordering::Relaxed), 1);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 80);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}
#[test]
#[ignore = "exclusive measured GPU; invalid replay must fail before writes and recover"]
fn source_qk_contract_replay_recovery_and_cpu_guards() {
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_QK_FIXTURES").unwrap());
    let d = root.join("layer3-m8-position2048-frequency0");
    let a = fixture(&d, 1);
    let expected = [u16s(&d.join("q.bf16")), u16s(&d.join("k.bf16"))];
    let key = 0x73716b726f706532;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut b = Builder(AtomicUsize::new(0));
    assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
    for rows in [1, 7, 33] {
        let mut bad = a.clone();
        bad[0] =
            MxArray::from_bfloat16(&vec![0; rows * 16 * 256], &[1, rows as i64, 16, 256]).unwrap();
        bad[1] =
            MxArray::from_bfloat16(&vec![0; rows * 2 * 256], &[1, rows as i64, 2, 256]).unwrap();
        assert!(raw(&bad).iter().all(|p| p.is_null()));
        assert!(compiled(&mut b, &bad, key).is_none());
        assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
    }
    assert_eq!(b.0.load(Ordering::Relaxed), 1);
    for j in 0..6 {
        let mut bad = a.clone();
        bad[j] = bad[j]
            .astype(if j < 2 {
                DType::Float32
            } else if j < 5 {
                DType::BFloat16
            } else {
                DType::Uint32
            })
            .unwrap();
        assert!(raw(&bad).iter().all(|p| p.is_null()));
        assert!(compiled(&mut b, &bad, key).is_none());
        let traces = b.0.load(Ordering::Relaxed);
        assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
        assert_eq!(b.0.load(Ordering::Relaxed), traces);
    }
    let mut bad = a.clone();
    bad[0] = MxArray::from_bfloat16(&vec![0; 8 * 16 * 256], &[1, 8, 256, 16])
        .unwrap()
        .transpose(Some(&[0, 1, 3, 2]))
        .unwrap();
    bad[0].eval();
    assert!(raw(&bad).iter().all(|p| p.is_null()));
    assert!(compiled(&mut b, &bad, key).is_none());
    let traces = b.0.load(Ordering::Relaxed);
    assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
    assert_eq!(b.0.load(Ordering::Relaxed), traces);
    let previous = unsafe { mlx_sys::mlx_default_device() };
    unsafe { mlx_sys::mlx_set_default_device(0) };
    let available = unsafe { mlx_sys::mlx_source_qk_norm_rope_available() };
    let rejected = raw(&a).iter().all(|p| p.is_null());
    unsafe { mlx_sys::mlx_set_default_device(previous) };
    assert!(!available && rejected);
    assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}
