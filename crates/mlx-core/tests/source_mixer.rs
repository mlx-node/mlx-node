//! Exact original Splash Q4 mixer arithmetic over lossless native T64 planes.
//! Fixtures use actual checkpoint weights and deterministic expanded captured
//! MoE inputs; they are not advertised as captured mixer activations.
use mlx_core::array::{DType, MxArray};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
const K: i64 = 4096;
const N: i64 = 2048;
const WORDS: usize = 16384 + 3072 + 32;
fn words16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn words32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn create(bytes: &[u8], dtype: DType, shape: &[i64]) -> MxArray {
    match dtype {
        DType::BFloat16 => MxArray::from_bfloat16(&words16(bytes), shape),
        DType::Float16 => MxArray::from_float16(&words16(bytes), shape),
        DType::Uint32 => MxArray::from_uint32(&words32(bytes), shape),
        DType::Uint8 => MxArray::from_uint8(bytes, shape),
        _ => panic!("unsupported fixture dtype"),
    }
    .unwrap()
}
fn load(path: &Path, dtype: DType, shape: &[i64], view: usize) -> MxArray {
    let bytes = std::fs::read(path).unwrap();
    let item = match dtype {
        DType::Uint8 => 1,
        DType::Uint32 => 4,
        _ => 2,
    };
    assert_eq!(bytes.len(), shape.iter().product::<i64>() as usize * item);
    if view == 1 {
        // Deliberately contiguous but not 16-byte aligned, on EVERY plane.
        let mut padded = vec![0; item];
        padded.extend_from_slice(&bytes);
        let len = padded.len() / item;
        let base = create(&padded, dtype, &[len as i64]);
        base.eval();
        let a = base
            .slice(&[1], &[len as i64])
            .unwrap()
            .reshape(shape)
            .unwrap();
        a.eval();
        assert_ne!(
            unsafe { mlx_sys::mlx_array_get_buffer_offset(a.as_raw_ptr()) } % 16,
            0
        );
        a
    } else if view == 2 {
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let mut transposed = vec![0; bytes.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[(c * rows + r) * item..(c * rows + r + 1) * item]
                    .copy_from_slice(&bytes[(r * cols + c) * item..(r * cols + c + 1) * item]);
            }
        }
        create(&transposed, dtype, &[cols as i64, rows as i64])
            .transpose(Some(&[1, 0]))
            .unwrap()
    } else {
        create(&bytes, dtype, shape)
    }
}
fn fixture(d: &Path, view: usize, residual: bool) -> Vec<MxArray> {
    let mut a = vec![
        load(&d.join("x.bf16"), DType::BFloat16, &[8, K], view),
        load(&d.join("weight.u32"), DType::Uint32, &[N, K / 8], view),
        load(&d.join("scales.u8"), DType::Uint8, &[N, K / 16], view),
        load(&d.join("bases.f16"), DType::Float16, &[N, K / 128], view),
    ];
    if residual {
        a.push(load(
            &d.join("residual.bf16"),
            DType::BFloat16,
            &[8, N],
            view,
        ));
    }
    a
}
fn raw(a: &[MxArray]) -> *mut mlx_sys::mlx_array {
    unsafe {
        mlx_sys::mlx_source_mixer_q4(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
            a[3].as_raw_ptr(),
            a.get(4).map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
        )
    }
}
fn eval(mut out: *mut mlx_sys::mlx_array, prepare: bool) -> Option<Vec<u32>> {
    if out.is_null() {
        return None;
    }
    let mut error = [0i8; 1024];
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut out, 1, error.as_mut_ptr(), error.len()) };
    let result = if !ok {
        None
    } else if prepare {
        assert_eq!(unsafe { mlx_sys::mlx_array_size(out) }, WORDS);
        let mut v = vec![0; WORDS];
        assert!(unsafe { mlx_sys::mlx_array_to_uint32(out, v.as_mut_ptr(), v.len()) });
        Some(v)
    } else {
        assert_eq!(unsafe { mlx_sys::mlx_array_size(out) }, (8 * N) as usize);
        let mut v = vec![0u16; (8 * N) as usize];
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(out, v.as_mut_ptr(), v.len()) });
        Some(v.into_iter().map(u32::from).collect())
    };
    unsafe { mlx_sys::mlx_array_delete(out) };
    result
}
struct Builder {
    prepare: bool,
    traces: AtomicUsize,
}
struct OwnedInputs<'a>(&'a [*const mlx_sys::mlx_array]);
impl Drop for OwnedInputs<'_> {
    fn drop(&mut self) {
        for &handle in self.0 {
            if !handle.is_null() {
                unsafe { mlx_sys::mlx_array_delete(handle.cast_mut()) };
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
    // The tracing bridge transfers ownership even when construction fails.
    let _owned_inputs = OwnedInputs(a);
    if ctx.is_null() || outputs.is_null() || m != 1 {
        return false;
    }
    let b = unsafe { &*ctx.cast::<Builder>() };
    if (b.prepare && n != 1) || (!b.prepare && n != 4 && n != 5) {
        return false;
    }
    b.traces.fetch_add(1, Ordering::Relaxed);
    let out = unsafe {
        if b.prepare {
            mlx_sys::mlx_test_source_mixer_q4_prepare(a[0].cast_mut())
        } else {
            mlx_sys::mlx_source_mixer_q4(
                a[0].cast_mut(),
                a[1].cast_mut(),
                a[2].cast_mut(),
                a[3].cast_mut(),
                a.get(4).map_or(std::ptr::null_mut(), |p| p.cast_mut()),
            )
        }
    };
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
        }
        return None;
    }
    eval(out, b.prepare)
}
fn assert_f64(d: &Path, got: &[u32], residual: bool) {
    let indices = words32(&std::fs::read(d.join("oracle-indices.u32")).unwrap());
    let reference: Vec<_> = std::fs::read(d.join("oracle.f64"))
        .unwrap()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let res = words16(&std::fs::read(d.join("residual.bf16")).unwrap());
    assert_eq!(indices.len(), 256);
    assert_eq!(reference.len(), 256);
    let (mut peak, mut max, mut ss, mut rr) = (0.0f64, 0.0f64, 0.0, 0.0);
    for (i, &v) in indices.iter().zip(&reference) {
        let i = *i as usize;
        let ref_value = v + if residual {
            f64::from(half::bf16::from_bits(res[i]).to_f32())
        } else {
            0.0
        };
        let value = f64::from(half::bf16::from_bits(got[i] as u16).to_f32());
        assert!(value.is_finite());
        let error = value - ref_value;
        peak = peak.max(ref_value.abs());
        max = max.max(error.abs());
        ss += error * error;
        rr += ref_value * ref_value;
    }
    assert!(
        max / peak < 0.004 && (ss / rr).sqrt() < 0.003,
        "F64 peak={} RMS={}",
        max / peak,
        (ss / rr).sqrt()
    );
}
#[test]
#[ignore = "exclusive measured M5 GPU; SPLASH_SOURCE_MIXER_FIXTURES points to original-source oracle"]
fn source_mixer_original_bits_prepare_f64_and_one_trace_replay() {
    assert!(unsafe { mlx_sys::mlx_source_mixer_q4_available() });
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_MIXER_FIXTURES").unwrap());
    for residual in [false, true] {
        let key = 0x736d697865723000 + u64::from(residual);
        let pkey = key + 16;
        unsafe {
            mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key);
            mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, pkey)
        };
        let mut b = Builder {
            prepare: false,
            traces: AtomicUsize::new(0),
        };
        let mut pb = Builder {
            prepare: true,
            traces: AtomicUsize::new(0),
        };
        for (iteration, layer) in [0, 5, 10, 39, 0].into_iter().enumerate() {
            let d = root.join(format!("layer{layer}"));
            let a = fixture(&d, iteration % 3, residual);
            let expected: Vec<u32> = words16(
                &std::fs::read(d.join(if residual {
                    "source-residual.bf16"
                } else {
                    "source-plain.bf16"
                }))
                .unwrap(),
            )
            .into_iter()
            .map(u32::from)
            .collect();
            let got = eval(raw(&a), false).unwrap();
            assert_eq!(got, expected, "source layer{layer}, residual{residual}");
            assert_f64(&d, &got, residual);
            for _ in 0..2 {
                assert_eq!(compiled(&mut b, &a, key).unwrap(), expected);
            }
            assert_eq!(b.traces.load(Ordering::Relaxed), 1);
            let mut prepared = words32(&std::fs::read(d.join("source-table.bf16")).unwrap());
            prepared.extend(words32(&std::fs::read(d.join("source-sums.f32")).unwrap()));
            prepared.extend([0; 32]);
            assert_eq!(prepared.len(), WORDS);
            assert_eq!(
                eval(
                    unsafe { mlx_sys::mlx_test_source_mixer_q4_prepare(a[0].as_raw_ptr()) },
                    true
                )
                .unwrap(),
                prepared
            );
            assert_eq!(compiled(&mut pb, &a[..1], pkey).unwrap(), prepared);
            assert_eq!(pb.traces.load(Ordering::Relaxed), 1);
            if iteration == 0 {
                for rows in [1, 7, 33] {
                    let mut invalid = a.clone();
                    invalid[0] = MxArray::zeros(&[rows, K], Some(DType::BFloat16)).unwrap();
                    if residual {
                        invalid[4] = MxArray::zeros(&[rows, N], Some(DType::BFloat16)).unwrap();
                    }
                    assert!(raw(&invalid).is_null());
                    assert!(compiled(&mut b, &invalid, key).is_none());
                }
                assert_eq!(
                    compiled(&mut b, &a, key).unwrap(),
                    expected,
                    "valid replay after rejected width"
                );
                assert_eq!(b.traces.load(Ordering::Relaxed), 1);
            }
        }
        unsafe {
            mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key);
            mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, pkey)
        };
    }
}
#[test]
#[ignore = "exclusive measured M5 GPU; explicit unsupported dtype/device guards"]
fn source_mixer_contract_and_cpu_guards() {
    let root = std::path::PathBuf::from(std::env::var("SPLASH_SOURCE_MIXER_FIXTURES").unwrap());
    let a = fixture(&root.join("layer0"), 0, true);
    for j in 0..a.len() {
        let mut bad = a.clone();
        bad[j] = bad[j].astype(DType::Float32).unwrap();
        assert!(raw(&bad).is_null(), "plane{j}");
    }
    let previous = unsafe { mlx_sys::mlx_default_device() };
    unsafe { mlx_sys::mlx_set_default_device(0) };
    let available = unsafe { mlx_sys::mlx_source_mixer_q4_available() };
    let rejected = raw(&a).is_null();
    unsafe { mlx_sys::mlx_set_default_device(previous) };
    assert!(!available && rejected);
    assert!(unsafe { mlx_sys::mlx_source_mixer_q4_available() });
}
