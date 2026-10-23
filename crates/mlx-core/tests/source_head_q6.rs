//! Original-source F32 head oracle, actual Q6 weights, shared fixture storage.
//! Inputs are synthetic or captured MoE inputs, not final normalized head inputs.
use mlx_core::array::{DType, MxArray};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
const K: i64 = 2048;
const N: i64 = 248320;
fn words16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn words32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn create(b: &[u8], dtype: DType, shape: &[i64]) -> MxArray {
    match dtype {
        DType::BFloat16 => MxArray::from_bfloat16(&words16(b), shape),
        DType::Float16 => MxArray::from_float16(&words16(b), shape),
        DType::Uint32 => MxArray::from_uint32(&words32(b), shape),
        DType::Int8 => MxArray::from_int8(&b.iter().map(|v| *v as i8).collect::<Vec<_>>(), shape),
        _ => panic!("fixture dtype"),
    }
    .unwrap()
}
fn load(
    path: &Path,
    dtype: DType,
    shape: &[i64],
    view: usize,
    swap: Option<(usize, usize)>,
) -> MxArray {
    let mut bytes = std::fs::read(path).unwrap();
    let item = match dtype {
        DType::Int8 => 1,
        DType::Uint32 => 4,
        _ => 2,
    };
    assert_eq!(bytes.len(), shape.iter().product::<i64>() as usize * item);
    if let Some((units, unit_bytes)) = swap {
        // Swap rows0/1 within every T64 tile, preserving its source K walk.
        // All code, signed-scale, and base inputs change on graph replay.
        for tile in 0..N as usize / 64 {
            for unit in 0..units {
                let at = (tile * units * 64 + unit * 64) * unit_bytes;
                let (left, right) = bytes[at..at + 2 * unit_bytes].split_at_mut(unit_bytes);
                left.swap_with_slice(right);
            }
        }
    }
    if view == 1 {
        let mut padded = vec![0; item];
        padded.extend_from_slice(&bytes);
        let len = padded.len() / item;
        let base = create(&padded, dtype, &[len as i64]);
        base.eval();
        let out = base
            .slice(&[1], &[len as i64])
            .unwrap()
            .reshape(shape)
            .unwrap();
        out.eval();
        assert_ne!(
            unsafe { mlx_sys::mlx_array_get_buffer_offset(out.as_raw_ptr()) } % 16,
            0
        );
        out
    } else if view == 2 {
        assert_eq!(shape.len(), 2);
        let (r, c) = (shape[0] as usize, shape[1] as usize);
        let mut t = vec![0; bytes.len()];
        for row in 0..r {
            for col in 0..c {
                t[(col * r + row) * item..(col * r + row + 1) * item]
                    .copy_from_slice(&bytes[(row * c + col) * item..(row * c + col + 1) * item]);
            }
        }
        create(&t, dtype, &[c as i64, r as i64])
            .transpose(Some(&[1, 0]))
            .unwrap()
    } else {
        create(&bytes, dtype, shape)
    }
}
fn fixture(root: &Path, case: usize, rank: usize) -> Vec<MxArray> {
    let swap = case == 1;
    let view = case;
    let mut x = load(
        &root.join(format!("case{case}/x.bf16")),
        DType::BFloat16,
        &[8, K],
        view,
        None,
    );
    if rank == 3 {
        x = x.reshape(&[1, 8, K]).unwrap();
    }
    vec![
        x,
        load(
            &root.join("weight.u32"),
            DType::Uint32,
            &[N, 384],
            if view == 2 { 0 } else { view },
            swap.then_some((64, 24)),
        ),
        load(
            &root.join("scales.i8"),
            DType::Int8,
            &[N, 128],
            if view == 2 { 0 } else { view },
            swap.then_some((8, 16)),
        ),
        load(
            &root.join("bases.f16"),
            DType::Float16,
            &[N, 8],
            if view == 2 { 0 } else { view },
            swap.then_some((8, 2)),
        ),
    ]
}
fn raw(a: &[MxArray]) -> *mut mlx_sys::mlx_array {
    unsafe {
        mlx_sys::mlx_source_head_q6(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
            a[3].as_raw_ptr(),
        )
    }
}
fn evaluate(mut output: *mut mlx_sys::mlx_array, a: &[MxArray]) -> Option<Vec<u32>> {
    if output.is_null() {
        return None;
    }
    let mut error = [0i8; 2048];
    let ok =
        unsafe { mlx_sys::mlx_eval_with_error(&mut output, 1, error.as_mut_ptr(), error.len()) };
    let result = if ok {
        assert_eq!(
            unsafe { mlx_sys::mlx_array_ndim(output) },
            a[0].shape().unwrap().len()
        );
        let mut shape = vec![0i64; unsafe { mlx_sys::mlx_array_ndim(output) }];
        unsafe { mlx_sys::mlx_array_shape(output, shape.as_mut_ptr()) };
        let mut want = a[0].shape().unwrap().as_ref().to_vec();
        *want.last_mut().unwrap() = N;
        assert_eq!(shape, want);
        let buffer = unsafe { mlx_sys::mlx_array_get_metal_buffer(output) };
        assert!(!buffer.is_null());
        for input in a {
            assert_ne!(buffer, unsafe {
                mlx_sys::mlx_array_get_metal_buffer(input.as_raw_ptr())
            });
        }
        let mut floats = vec![0f32; (8 * N) as usize];
        assert_eq!(unsafe { mlx_sys::mlx_array_size(output) }, floats.len());
        assert!(unsafe {
            mlx_sys::mlx_array_to_float32(output, floats.as_mut_ptr(), floats.len())
        });
        assert!(floats.iter().all(|v| v.is_finite()));
        Some(floats.into_iter().map(f32::to_bits).collect())
    } else {
        None
    };
    unsafe { mlx_sys::mlx_array_delete(output) };
    result
}
struct Builder(AtomicUsize);
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
    if ctx.is_null() || outputs.is_null() || n != 4 || m != 1 {
        return false;
    }
    unsafe { &*ctx.cast::<Builder>() }
        .0
        .fetch_add(1, Ordering::Relaxed);
    let out = unsafe {
        mlx_sys::mlx_source_head_q6(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
            a[3].cast_mut(),
        )
    };
    unsafe { *outputs = out };
    !out.is_null()
}
fn compiled(b: &mut Builder, a: &[MxArray], key: u64) -> Option<Vec<u32>> {
    let pointers: Vec<_> = a.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
    let mut out = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_compiled_graph_invoke(
            key,
            Some(build),
            (b as *mut Builder).cast(),
            pointers.as_ptr(),
            4,
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
fn source_expected(root: &Path, case: usize) -> Vec<u32> {
    let mut v = words32(&std::fs::read(root.join(format!("case{case}/source.f32"))).unwrap());
    assert_eq!(v.len(), (8 * N) as usize);
    if case == 1 {
        for row in 0..8 {
            for tile in 0..N as usize / 64 {
                v.swap(
                    row * N as usize + tile * 64,
                    row * N as usize + tile * 64 + 1,
                );
            }
        }
    }
    v
}
fn exact(got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    if got != want {
        let first: Vec<_> = got
            .iter()
            .zip(want)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(8)
            .map(|(i, (&a, &b))| (i, a, b))
            .collect();
        let count = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!("source F32 mismatches={count}, first={first:?}");
    }
}
fn f64_gate(root: &Path, case: usize, actual: &[u32]) {
    let dir = root.join(format!("case{case}"));
    let indices = words32(&std::fs::read(dir.join("oracle-indices.u32")).unwrap());
    for (name, limit) in [("oracle-stage.f64", 1e-5), ("oracle-exact.f64", 0.001)] {
        let refs: Vec<_> = std::fs::read(dir.join(name))
            .unwrap()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(refs.len(), indices.len());
        let (mut peak, mut error) = (0f64, 0f64);
        for (&index, &reference) in indices.iter().zip(&refs) {
            let mut index = index as usize;
            if case == 1 {
                let col = index % N as usize;
                index = if col % 64 == 0 {
                    index + 1
                } else if col % 64 == 1 {
                    index - 1
                } else {
                    index
                };
            }
            peak = peak.max(reference.abs());
            error = error.max((f32::from_bits(actual[index]) as f64 - reference).abs());
        }
        assert!(error / peak < limit, "{name} peak={}", error / peak);
    }
}
#[test]
#[ignore = "exclusive measured GPU; shared actual head fixture and original-source outputs"]
fn source_head_q6_original_bits_f64_and_dynamic_weights_replay() {
    assert!(unsafe { mlx_sys::mlx_source_head_q6_available() });
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_HEAD_FIXTURES").unwrap());
    for rank in [2, 3] {
        let key = 0x7371686561640000 + rank as u64;
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        let mut b = Builder(AtomicUsize::new(0));
        for case in [0, 1, 2, 0] {
            let a = fixture(&root, case, rank);
            let expected = source_expected(&root, case);
            let eager = evaluate(raw(&a), &a).unwrap();
            exact(&eager, &expected);
            f64_gate(&root, case, &eager);
            exact(&compiled(&mut b, &a, key).unwrap(), &expected);
            assert_eq!(b.0.load(Ordering::Relaxed), 1);
        }
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}
#[test]
#[ignore = "exclusive measured GPU; fail-closed shape/dtype replay and independent output"]
fn source_head_q6_contract_recovery_and_cpu_guards() {
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_HEAD_FIXTURES").unwrap());
    let a = fixture(&root, 0, 3);
    let expected = source_expected(&root, 0);
    let key = 0x7371686561640100;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut b = Builder(AtomicUsize::new(0));
    exact(&compiled(&mut b, &a, key).unwrap(), &expected);
    for rows in [1, 7, 33] {
        let mut bad = a.clone();
        bad[0] = MxArray::from_bfloat16(&vec![0; rows * 2048], &[1, rows as i64, K]).unwrap();
        assert!(raw(&bad).is_null());
        assert!(compiled(&mut b, &bad, key).is_none());
        exact(&compiled(&mut b, &a, key).unwrap(), &expected);
    }
    assert_eq!(b.0.load(Ordering::Relaxed), 1);
    for j in 0..4 {
        let mut bad = a.clone();
        bad[j] = bad[j].astype(DType::Float32).unwrap();
        assert!(raw(&bad).is_null());
        assert!(compiled(&mut b, &bad, key).is_none());
        let traces = b.0.load(Ordering::Relaxed);
        exact(&compiled(&mut b, &a, key).unwrap(), &expected);
        assert_eq!(b.0.load(Ordering::Relaxed), traces);
    }
    // Packed planes are opaque T64 byte streams. Positive strides do not
    // turn a transposed/padded logical matrix into a supported packed layout.
    let mut bad = a.clone();
    bad[2] = load(&root.join("scales.i8"), DType::Int8, &[N, 128], 2, None);
    bad[2].eval();
    assert!(raw(&bad).is_null());
    assert!(compiled(&mut b, &bad, key).is_none());
    exact(&compiled(&mut b, &a, key).unwrap(), &expected);
    let before = unsafe { mlx_sys::mlx_default_device() };
    unsafe { mlx_sys::mlx_set_default_device(0) };
    let available = unsafe { mlx_sys::mlx_source_head_q6_available() };
    let rejected = raw(&a).is_null();
    unsafe { mlx_sys::mlx_set_default_device(before) };
    assert!(!available && rejected);
    exact(&compiled(&mut b, &a, key).unwrap(), &expected);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}
