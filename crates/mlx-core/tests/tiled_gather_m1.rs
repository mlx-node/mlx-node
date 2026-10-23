//! Baseline one-token gather arithmetic on the sole T64 expert allocation.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Kept explicit so this artifact can be typechecked against the frozen V15
// Rust library before its new C ABI is linked by the next canonical build.
unsafe extern "C" {
    fn mlx_tiled_gather_qmv_m1(
        x: *mut mlx_sys::mlx_array,
        w: *mut mlx_sys::mlx_array,
        scales: *mut mlx_sys::mlx_array,
        biases: *mut mlx_sys::mlx_array,
        indices: *mut mlx_sys::mlx_array,
        bits: i32,
        output: *mut *mut mlx_sys::mlx_array,
    ) -> i32;
}
fn tiled(a: &MxArray, unit: i64) -> MxArray {
    let shape = a.shape().unwrap();
    kquant_tile_rows(&a.reshape(&[-1, shape[2]]).unwrap(), unit)
        .unwrap()
        .reshape(&shape)
        .unwrap()
}
fn call(x: &MxArray, w: &Weights, index: &MxArray, bits: i32) -> (i32, *mut mlx_sys::mlx_array) {
    let mut out = std::ptr::null_mut();
    let status = unsafe {
        mlx_tiled_gather_qmv_m1(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            index.as_raw_ptr(),
            bits,
            &mut out,
        )
    };
    (status, out)
}
fn reference(x: &MxArray, w: &Weights, index: &MxArray, q: &KQuant) -> *mut mlx_sys::mlx_array {
    let mode = std::ffi::CString::new(q.mode).unwrap();
    unsafe {
        mlx_sys::mlx_test_kquant_gather_qmm(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            std::ptr::null_mut(),
            index.as_raw_ptr(),
            true,
            q.group_size,
            q.bits,
            mode.as_ptr(),
            false,
            GPU,
        )
    }
}
fn offset(a: &MxArray) -> MxArray {
    let flat = a.reshape(&[-1]).unwrap();
    let zero = MxArray::zeros(&[1], Some(a.dtype().unwrap())).unwrap();
    MxArray::concatenate(&zero, &flat, 0)
        .unwrap()
        .slice(&[1], &[a.size().unwrap() as i64 + 1])
        .unwrap()
        .reshape(&a.shape().unwrap())
        .unwrap()
}
#[test]
fn tiled_m1_gather_preserves_baseline_bits_and_boundaries() {
    let _lock = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    for mode in ["q4k", "q5k", "q6k"] {
        let q = KQUANTS.iter().find(|q| q.mode == mode).unwrap();
        for (e, n, k) in [(3, 64, 512), (9, 512, 2048), (9, 2048, 512)] {
            let row = weights(q, &[e, n], k, 0x8129);
            let t = Weights {
                w: tiled(&row.w, i64::from(q.bits)),
                scales: tiled(&row.scales, 16),
                biases: tiled(&row.biases, if q.bits == 6 { 1 } else { 2 }),
            };
            for (case, ids) in [
                vec![0],
                vec![e as i32 - 1, 0, 1],
                vec![0, 1, 0, 1, e as i32 - 1, 1, 0, 0],
            ]
            .into_iter()
            .enumerate()
            {
                let index = MxArray::from_int32(&ids, &[1, ids.len() as i64]).unwrap();
                for broadcast in [true, false] {
                    let x = activation(
                        &[1, if broadcast { 1 } else { ids.len() as i64 }, 1, k],
                        0x9017 + case as u32,
                        DType::BFloat16,
                    );
                    let expected = read_bits("baseline gather", reference(&x, &row, &index, q));
                    for misaligned in [false, true] {
                        let (xm, tm, im) = if misaligned {
                            (
                                offset(&x),
                                Weights {
                                    w: offset(&t.w),
                                    scales: offset(&t.scales),
                                    biases: offset(&t.biases),
                                },
                                offset(&index),
                            )
                        } else {
                            (
                                x.clone(),
                                Weights {
                                    w: t.w.clone(),
                                    scales: t.scales.clone(),
                                    biases: t.biases.clone(),
                                },
                                index.clone(),
                            )
                        };
                        start_counting();
                        let (status, out) = call(&xm, &tm, &im, q.bits);
                        assert_eq!(status, 1);
                        let actual = read_bits("T64 gather", out);
                        assert_eq!(family_count("gather_t64_m1"), 1);
                        stop_counting();
                        assert_eq!(
                            actual, expected,
                            "{mode} E{e} N{n} K{k} broadcast{broadcast} misaligned{misaligned}"
                        );
                    }
                }
            }
            // Invalid device indices cannot read another expert allocation.
            let invalid = MxArray::from_int32(&[0, -1, e as i32], &[1, 3]).unwrap();
            let x = activation(&[1, 1, 1, k], 311, DType::BFloat16);
            let (status, out) = call(&x, &t, &invalid, q.bits);
            assert_eq!(status, 1);
            let actual = read_bits("bounded T64 gather", out).2;
            let valid = MxArray::from_int32(&[0], &[1, 1]).unwrap();
            let expected = read_bits("valid baseline lane", reference(&x, &row, &valid, q)).2;
            assert_eq!(&actual[..n as usize], expected.as_slice());
            assert!(actual[n as usize..].iter().all(|&v| v == 0));
            let index = MxArray::from_int32(&[0], &[1, 1]).unwrap();
            for x in [
                activation(&[2, 1, 1, k], 3, DType::BFloat16),
                activation(&[1, 1, 2, k], 3, DType::BFloat16),
                activation(&[1, 1, 1, k], 3, DType::Float32),
            ] {
                let (status, out) = call(&x, &t, &index, q.bits);
                assert_eq!(status, 0);
                assert!(out.is_null());
            }
        }
    }
}
struct Trace {
    bits: i32,
    builds: std::sync::atomic::AtomicUsize,
}
unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    count: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || count != 5 || n_outputs != 1 {
        return false;
    }
    let ctx = unsafe { &*context.cast::<Trace>() };
    ctx.builds
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let a = unsafe { std::slice::from_raw_parts(inputs, count) };
    let result = unsafe {
        mlx_tiled_gather_qmv_m1(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
            a[3].cast_mut(),
            a[4].cast_mut(),
            ctx.bits,
            outputs,
        )
    };
    for &h in a {
        unsafe { mlx_sys::mlx_array_delete(h.cast_mut()) };
    }
    result == 1
}
#[test]
fn tiled_m1_gather_replays_changed_experts_weights_and_geometry() {
    let _lock = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    for mode in ["q4k", "q5k", "q6k"] {
        let q = KQUANTS.iter().find(|q| q.mode == mode).unwrap();
        let key = 0x6A7E_0000_0000_0000 + q.bits as u64;
        let mut trace = Trace {
            bits: q.bits,
            builds: std::sync::atomic::AtomicUsize::new(0),
        };
        for (i, (e, n, k, xrows, count)) in [
            (3, 64, 512, 1, 3),
            (9, 512, 2048, 8, 8),
            (9, 2048, 512, 1, 8),
            (4, 128, 1024, 3, 3),
        ]
        .into_iter()
        .enumerate()
        {
            let row = weights(q, &[e, n], k, 0x721 + i as u32 * 31);
            let t = Weights {
                w: tiled(&row.w, i64::from(q.bits)),
                scales: tiled(&row.scales, 16),
                biases: tiled(&row.biases, if q.bits == 6 { 1 } else { 2 }),
            };
            let ids = (0..count)
                .map(|j| ((j + i as i64) % e) as i32)
                .collect::<Vec<_>>();
            let index = MxArray::from_int32(&ids, &[1, count]).unwrap();
            let x = activation(&[1, xrows, 1, k], 41 + i as u32 * 19, DType::BFloat16);
            let a = [&x, &t.w, &t.scales, &t.biases, &index];
            let inputs = a.map(|a| a.as_raw_ptr().cast_const());
            let mut output = std::ptr::null_mut();
            assert!(unsafe {
                mlx_sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build),
                    (&mut trace as *mut Trace).cast(),
                    inputs.as_ptr(),
                    5,
                    &mut output,
                    1,
                    true,
                )
            });
            assert_eq!(
                read_bits("T64 replay", output),
                read_bits("baseline replay", reference(&x, &row, &index, q)),
                "{mode} replay{i}"
            );
        }
        assert_eq!(trace.builds.load(std::sync::atomic::Ordering::Relaxed), 1);
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}
