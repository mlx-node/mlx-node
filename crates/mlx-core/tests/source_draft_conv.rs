//! Original Splash prepare/finish convolution bits. Actual companion BF16 base
//! kernels and deterministic finite operands; not captured draft activations.
use mlx_core::array::{DType, MxArray};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
fn words(path: &Path) -> Vec<u16> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn load(path: &Path, shape: &[i64], view: usize) -> MxArray {
    let values = words(path);
    assert_eq!(values.len(), shape.iter().product::<i64>() as usize);
    if view == 0 {
        return MxArray::from_bfloat16(&values, shape).unwrap();
    }
    if view == 1 {
        let mut padded = vec![0x7fc1];
        padded.extend(&values);
        let base = MxArray::from_bfloat16(&padded, &[padded.len() as i64]).unwrap();
        base.eval();
        let a = base
            .slice(&[1], &[padded.len() as i64])
            .unwrap()
            .reshape(shape)
            .unwrap();
        a.eval();
        assert_eq!(
            unsafe { mlx_sys::mlx_array_get_buffer_offset(a.as_raw_ptr()) },
            2
        );
        return a;
    }
    let mut big = shape.to_vec();
    for n in &mut big {
        if *n > 1 {
            *n += 2;
        }
    }
    let mut padded = vec![0x7fc1; big.iter().product::<i64>() as usize];
    for (i, &v) in values.iter().enumerate() {
        let mut rem = i;
        let mut index = 0usize;
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            let c = rem % shape[axis] as usize;
            rem /= shape[axis] as usize;
            index += (c + usize::from(shape[axis] > 1)) * stride;
            stride *= big[axis] as usize;
        }
        padded[index] = v;
    }
    let base = MxArray::from_bfloat16(&padded, &big).unwrap();
    base.eval();
    let starts: Vec<_> = shape.iter().map(|n| i64::from(*n > 1)).collect();
    let stops: Vec<_> = starts.iter().zip(shape).map(|(s, n)| s + n).collect();
    let a = base.slice(&starts, &stops).unwrap();
    a.eval();
    assert_ne!(
        unsafe { mlx_sys::mlx_array_get_buffer_offset(a.as_raw_ptr()) },
        0
    );
    a
}
fn fixture(d: &Path, side: i32, view: usize) -> Vec<MxArray> {
    let mut a = vec![
        load(&d.join("x.bf16"), &[1, 8, 2048], view),
        load(&d.join("dynamic.bf16"), &[1, 8, 2, 2, 128], view),
        load(&d.join("base.bf16"), &[2, 2, 2048], view),
    ];
    if side == 1 {
        a.push(load(&d.join("residual.bf16"), &[1, 8, 2048], view));
    }
    a
}
fn raw(a: &[MxArray], side: i32) -> *mut mlx_sys::mlx_array {
    unsafe {
        mlx_sys::mlx_source_dflash2_conv(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
            side,
            a.get(3).map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
        )
    }
}
fn evaluate(mut out: *mut mlx_sys::mlx_array, inputs: &[MxArray]) -> Option<Vec<u16>> {
    if out.is_null() {
        return None;
    }
    let mut err = [0i8; 2048];
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut out, 1, err.as_mut_ptr(), err.len()) };
    let got = if ok {
        let buf = unsafe { mlx_sys::mlx_array_get_metal_buffer(out) };
        assert!(!buf.is_null());
        for a in inputs {
            assert_ne!(buf, unsafe {
                mlx_sys::mlx_array_get_metal_buffer(a.as_raw_ptr())
            });
        }
        let mut v = vec![0u16; 16384];
        assert_eq!(unsafe { mlx_sys::mlx_array_size(out) }, v.len());
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(out, v.as_mut_ptr(), v.len()) });
        Some(v)
    } else {
        None
    };
    unsafe { mlx_sys::mlx_array_delete(out) };
    got
}
struct Builder {
    side: i32,
    traces: AtomicUsize,
}
struct Owned<'a>(&'a [*const mlx_sys::mlx_array]);
impl Drop for Owned<'_> {
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
    let _owned = Owned(a);
    if ctx.is_null() || outputs.is_null() || m != 1 {
        return false;
    }
    let b = unsafe { &*ctx.cast::<Builder>() };
    if n != if b.side == 1 { 4 } else { 3 } {
        return false;
    }
    b.traces.fetch_add(1, Ordering::Relaxed);
    let out = unsafe {
        mlx_sys::mlx_source_dflash2_conv(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
            b.side,
            a.get(3).map_or(std::ptr::null_mut(), |p| p.cast_mut()),
        )
    };
    unsafe { *outputs = out };
    !out.is_null()
}
fn compiled(b: &mut Builder, a: &[MxArray], key: u64) -> Option<Vec<u16>> {
    let pointers: Vec<_> = a.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
    let mut out = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_compiled_graph_invoke(
            key,
            Some(build),
            (b as *mut Builder).cast(),
            pointers.as_ptr(),
            pointers.len(),
            &mut out,
            1,
            true,
        )
    };
    if !ok {
        if !out.is_null() {
            unsafe { mlx_sys::mlx_array_delete(out) }
        }
        return None;
    }
    evaluate(out, a)
}
fn f64_gate(d: &Path, side: i32, got: &[u16]) {
    let refs: Vec<_> = std::fs::read(d.join(format!("oracle-{side}.f64")))
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| f64::from_le_bytes(*x))
        .collect();
    assert_eq!(refs.len(), got.len());
    let (mut peak, mut max, mut ss, mut rr) = (0f64, 0f64, 0f64, 0f64);
    for (&b, &r) in got.iter().zip(&refs) {
        let v = half::bf16::from_bits(b).to_f32() as f64;
        assert!(v.is_finite());
        let error = v - r;
        peak = peak.max(r.abs());
        max = max.max(error.abs());
        ss += error * error;
        rr += r * r;
    }
    assert!(
        max / peak < 0.004 && (ss / rr).sqrt() < 0.003,
        "F64 peak={} rms={}",
        max / peak,
        (ss / rr).sqrt()
    );
}
#[test]
#[ignore = "exclusive GPU; SPLASH_SOURCE_DRAFT_CONV_FIXTURES original production oracle"]
fn source_draft_conv_original_bits_f64_and_one_trace() {
    assert!(unsafe { mlx_sys::mlx_source_dflash2_conv_available() });
    let root =
        std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_DRAFT_CONV_FIXTURES").unwrap());
    for side in [0, 1] {
        let key = 0x7364636f6e760000 + side as u64;
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        let mut b = Builder {
            side,
            traces: AtomicUsize::new(0),
        };
        let mut cases = 0;
        for layer in 0..6 {
            for kind in ["attention", "mlp"] {
                for seed in [0, 1] {
                    let d = root.join(format!("layer{layer}-{kind}-seed{seed}"));
                    let a = fixture(&d, side, cases % 3);
                    let expected = words(&d.join(format!("source-{side}.bf16")));
                    let eager = evaluate(raw(&a, side), &a).unwrap();
                    assert_eq!(eager, expected, "{d:?} side{side}");
                    f64_gate(&d, side, &eager);
                    assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
                    assert_eq!(b.traces.load(Ordering::Relaxed), 1);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 24);
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}
#[test]
#[ignore = "exclusive GPU; selected invalid shapes and output meanings fail closed"]
fn source_draft_conv_contract_replay_recovery() {
    let root =
        std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_DRAFT_CONV_FIXTURES").unwrap());
    let d = root.join("layer0-attention-seed0");
    for side in [0, 1] {
        let a = fixture(&d, side, 2);
        let expected = words(&d.join(format!("source-{side}.bf16")));
        let key = 0x7364636f6e760100 + side as u64;
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        let mut b = Builder {
            side,
            traces: AtomicUsize::new(0),
        };
        assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
        for rows in [1, 7, 33] {
            let mut bad = a.clone();
            bad[0] =
                MxArray::from_bfloat16(&vec![0; rows * 2048], &[1, rows as i64, 2048]).unwrap();
            bad[1] =
                MxArray::from_bfloat16(&vec![0; rows * 512], &[1, rows as i64, 2, 2, 128]).unwrap();
            if side == 1 {
                bad[3] = bad[0].clone();
            }
            assert!(raw(&bad, side).is_null());
            assert!(compiled(&mut b, &bad, key).is_none());
            assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
        }
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        for j in 0..a.len() {
            let mut bad = a.clone();
            bad[j] = bad[j].astype(DType::Float32).unwrap();
            assert!(raw(&bad, side).is_null());
            assert!(compiled(&mut b, &bad, key).is_none());
            let traces = b.traces.load(Ordering::Relaxed);
            assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
            assert_eq!(b.traces.load(Ordering::Relaxed), traces);
        }
        let mut bad = a.clone();
        bad[0] = MxArray::from_bfloat16(&vec![0; 16384], &[1, 2048, 8])
            .unwrap()
            .transpose(Some(&[0, 2, 1]))
            .unwrap();
        bad[0].eval();
        assert!(raw(&bad, side).is_null());
        assert!(compiled(&mut b, &bad, key).is_none());
        assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
        assert!(raw(&a, 1 - side).is_null());
        assert!(raw(&a, -1).is_null());
        assert!(raw(&a, 2).is_null());
        let previous = unsafe { mlx_sys::mlx_default_device() };
        unsafe { mlx_sys::mlx_set_default_device(0) };
        let available = unsafe { mlx_sys::mlx_source_dflash2_conv_available() };
        let rejected = raw(&a, side).is_null();
        unsafe { mlx_sys::mlx_set_default_device(previous) };
        assert!(!available && rejected);
        assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}
