//! Producer-prepared native SG8: exact packing, recurrence and graph replay.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Inputs {
    arrays: Vec<MxArray>,
    state_dst: MxArray,
    history_dst: MxArray,
}
impl Inputs {
    fn new(rows: i64, heads: i64, strided: bool, seed: u32) -> Self {
        let width = 2 * 16 * 128 + heads * 128;
        let packed = activation(&[1, rows, width + heads * 130], seed, DType::BFloat16);
        let take = |begin, end| {
            let view = packed.slice(&[0, 0, begin], &[1, rows, end]).unwrap();
            if strided {
                view
            } else {
                activation(
                    &[1, rows, end - begin],
                    seed + begin as u32,
                    DType::BFloat16,
                )
            }
        };
        let qkv = take(0, width);
        let z = take(width, width + heads * 128);
        let a = take(width + heads * 128, width + heads * 129);
        let b = take(width + heads * 129, width + heads * 130);
        Self {
            arrays: vec![
                qkv,
                a,
                b,
                activation(&[width, 4], seed + 3, DType::Float32),
                activation(&[3, width], seed + 7, DType::BFloat16),
                MxArray::from_float32(&vec![-0.3; heads as usize], &[heads]).unwrap(),
                activation(&[heads], seed + 11, DType::Float32),
                activation(&[1, heads, 128, 128], seed + 13, DType::Float32),
                z,
                MxArray::ones(&[128], Some(DType::BFloat16)).unwrap(),
            ],
            state_dst: MxArray::zeros(&[1, heads, 128, 128], Some(DType::Float32)).unwrap(),
            history_dst: MxArray::zeros(&[3, width], Some(DType::BFloat16)).unwrap(),
        }
    }
    fn call(&self, destinations: bool, prepared: bool) -> Vec<*mut mlx_sys::mlx_array> {
        let a: Vec<_> = self.arrays.iter().map(MxArray::as_raw_ptr).collect();
        let mut out = vec![std::ptr::null_mut(); if prepared { 10 } else { 8 }];
        let launch = if prepared {
            mlx_sys::mlx_gdn_fused_complete_prepared
        } else {
            mlx_sys::mlx_gdn_fused_complete
        };
        assert!(unsafe {
            launch(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                1e-6,
                if destinations {
                    self.state_dst.as_raw_ptr()
                } else {
                    std::ptr::null_mut()
                },
                if destinations {
                    self.history_dst.as_raw_ptr()
                } else {
                    std::ptr::null_mut()
                },
                out.as_mut_ptr(),
            )
        });
        out
    }
}

fn packed_bits(handle: *mut mlx_sys::mlx_array, count: usize) -> Vec<u32> {
    let mut output = vec![0; count];
    assert!(unsafe { mlx_sys::mlx_array_to_uint32(handle, output.as_mut_ptr(), count) });
    unsafe { mlx_sys::mlx_array_delete(handle) };
    output
}

// Independent logical-to-MMA packing oracle. Each row's output scalar has
// one physical halfword; the reduction tree is the old prep kernel's tree.
fn expected_tables(output: &[u32], width: usize) -> (Vec<u32>, Vec<u32>) {
    let mut packed = vec![0u32; width * 4];
    let mut sums = vec![0u32; width / 32 * 8];
    for row in 0..8 {
        for span in 0..width / 32 {
            let mut partial = [0f32; 8];
            for fm in 0..8 {
                let fn_ = row & !1;
                let lane = ((fm & 4) << 2) | ((fm & 3) << 1) | ((fn_ & 4) << 1) | ((fn_ & 2) >> 1);
                let mut values = [0f32; 4];
                for j in 0..4 {
                    let bits = output[row * width + span * 32 + 4 * fm + j] << 16;
                    packed[(span * 32 + lane) * 4 + j] |= (bits >> 16) << (16 * (row & 1));
                    values[j] = f32::from_bits(bits);
                }
                partial[fm] = (values[0] + values[1]) + (values[2] + values[3]);
            }
            sums[span * 8 + row] = (((partial[0] + partial[1]) + (partial[2] + partial[3]))
                + ((partial[4] + partial[5]) + (partial[6] + partial[7])))
                .to_bits();
        }
    }
    (packed, sums)
}

fn check_outputs(input: &Inputs, mut actual: Vec<*mut mlx_sys::mlx_array>) {
    assert_eq!(actual.len(), 10);
    assert!(unsafe { mlx_sys::mlx_eval(actual.as_mut_ptr(), actual.len()) });
    let reference: Vec<_> = input
        .call(false, false)
        .into_iter()
        .map(|h| read_bits("ordinary GDN", h))
        .collect();
    let width = reference[0].0[2] as usize;
    let expected = expected_tables(&reference[0].2, width);
    let sums = read_bits("prepared sums", actual.pop().unwrap());
    let packed = packed_bits(actual.pop().unwrap(), width * 4);
    assert_eq!(
        packed, expected.0,
        "every packed BF16 pair must match native SG8 mapping"
    );
    assert_eq!(
        sums.2, expected.1,
        "every FP32 group sum must preserve native SG8 reduction order"
    );
    for (slot, (got, want)) in actual
        .into_iter()
        .map(|h| read_bits("prepared GDN", h))
        .zip(reference)
        .enumerate()
    {
        assert_eq!(
            got, want,
            "prepared producer changed ordinary output {slot}"
        );
    }
}

#[test]
fn gdn_prepared_sg8_preserves_outputs_pack_sums_and_aliases() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    for scan in ["0", "1"] {
        unsafe { std::env::set_var("MLX_GDN_SCAN8", scan) };
        for (heads, strided, seed) in [(32, false, 17), (32, true, 29), (48, true, 41)] {
            let input = Inputs::new(8, heads, strided, seed);
            let mut actual = input.call(true, true);
            assert!(unsafe { mlx_sys::mlx_eval(actual.as_mut_ptr(), actual.len()) });
            assert!(unsafe { mlx_sys::mlx_array_aliases(actual[1], input.state_dst.as_raw_ptr()) });
            assert!(unsafe {
                mlx_sys::mlx_array_aliases(actual[7], input.history_dst.as_raw_ptr())
            });
            check_outputs(&input, actual);
        }
    }
    unsafe { std::env::remove_var("MLX_GDN_SCAN8") };
}

unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    count: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if count != 10 || n_outputs != 10 || context.is_null() {
        return false;
    }
    unsafe { &*context.cast::<std::sync::atomic::AtomicUsize>() }
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, count) }
        .iter()
        .map(|a| a.cast_mut())
        .collect();
    let result = unsafe {
        mlx_sys::mlx_gdn_fused_complete_prepared(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            1e-6,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            outputs,
        )
    };
    for a in a {
        unsafe { mlx_sys::mlx_array_delete(a) };
    }
    result
}

#[test]
fn gdn_prepared_sg8_replays_new_activations_weights_and_strides() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    let key = 0x6DD8_0000_0000_00F1;
    let mut traces = std::sync::atomic::AtomicUsize::new(0);
    for (heads, strided, seed) in [
        (32, false, 101),
        (48, true, 113),
        (32, true, 127),
        (32, false, 139),
    ] {
        let input = Inputs::new(8, heads, strided, seed);
        let arrays: Vec<_> = input
            .arrays
            .iter()
            .map(|a| a.as_raw_ptr().cast_const())
            .collect();
        let mut output = vec![std::ptr::null_mut(); 10];
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut traces as *mut std::sync::atomic::AtomicUsize).cast(),
                arrays.as_ptr(),
                arrays.len(),
                output.as_mut_ptr(),
                output.len(),
                true,
            )
        });
        check_outputs(&input, output);
    }
    assert_eq!(traces.load(std::sync::atomic::Ordering::Relaxed), 1);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}

#[test]
fn gdn_prepared_sg8_projection_matches_fresh_packing() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    unsafe { std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "1") };
    let q = KQUANTS.iter().find(|q| q.mode == "q4k").unwrap();
    let row_major = weights(q, &[2048], 4096, 0xA721);
    let tiled = Weights {
        w: kquant_tile_rows(&row_major.w, 4).unwrap(),
        scales: kquant_tile_rows(&row_major.scales, 16).unwrap(),
        biases: kquant_tile_rows(&row_major.biases, 2).unwrap(),
    };
    for seed in [151, 163, 179] {
        let input = Inputs::new(8, 32, true, seed);
        let mut a = input.call(false, true);
        assert!(unsafe { mlx_sys::mlx_eval(a.as_mut_ptr(), a.len()) });
        let mut output = std::ptr::null_mut();
        start_counting();
        assert_eq!(
            unsafe {
                mlx_sys::mlx_kquant_sg8_prepared(
                    a[0],
                    tiled.w.as_raw_ptr(),
                    tiled.scales.as_raw_ptr(),
                    tiled.biases.as_raw_ptr(),
                    a[8],
                    a[9],
                    &mut output,
                )
            },
            1
        );
        let actual = read_bits("prepared projection", output);
        assert_eq!(family_count("qmv_sg8_t64_prepared"), 1);
        stop_counting();
        let mode = std::ffi::CString::new("q4k@t64").unwrap();
        let reference = unsafe {
            mlx_sys::mlx_test_kquant_quantized_matmul(
                a[0],
                tiled.w.as_raw_ptr(),
                tiled.scales.as_raw_ptr(),
                tiled.biases.as_raw_ptr(),
                true,
                32,
                4,
                mode.as_ptr(),
                GPU,
            )
        };
        assert_eq!(actual, read_bits("fresh packing projection", reference));
        for h in a {
            unsafe { mlx_sys::mlx_array_delete(h) };
        }
    }
    unsafe { std::env::remove_var("MLX_KQUANT_REGISTER_OUTPUT") };
}

#[test]
fn attention_prepared_sg8_projection_matches_fresh_packing() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    struct RestoreEnv(&'static str, Option<std::ffi::OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            unsafe {
                match &self.1 {
                    Some(value) => std::env::set_var(self.0, value),
                    None => std::env::remove_var(self.0),
                }
            }
        }
    }
    let _register = RestoreEnv(
        "MLX_KQUANT_REGISTER_OUTPUT",
        std::env::var_os("MLX_KQUANT_REGISTER_OUTPUT"),
    );
    let _split = RestoreEnv(
        "MLX_KQUANT_SG8_SPLIT_K",
        std::env::var_os("MLX_KQUANT_SG8_SPLIT_K"),
    );
    unsafe { std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "1") };
    let q = KQUANTS.iter().find(|q| q.mode == "q4k").unwrap();
    let row_major = weights(q, &[2048], 4096, 0xA891);
    let tiled = Weights {
        w: kquant_tile_rows(&row_major.w, 4).unwrap(),
        scales: kquant_tile_rows(&row_major.scales, 16).unwrap(),
        biases: kquant_tile_rows(&row_major.biases, 2).unwrap(),
    };
    let mode = std::ffi::CString::new("q4k@t64").unwrap();
    for split in ["0", "1"] {
        unsafe { std::env::set_var("MLX_KQUANT_SG8_SPLIT_K", split) };
        for seed in [191, 211, 223] {
            let gate = activation(&[1, 8, 4096], seed, DType::BFloat16);
            let attention = activation(&[1, 16, 8, 256], seed + 7, DType::BFloat16);
            let mut a = [std::ptr::null_mut(); 3];
            assert_eq!(
                unsafe {
                    mlx_sys::mlx_attention_sg8_prepare(
                        gate.as_raw_ptr(),
                        attention.as_raw_ptr(),
                        &mut a[0],
                        &mut a[1],
                        &mut a[2],
                    )
                },
                1
            );
            assert!(unsafe { mlx_sys::mlx_eval(a.as_mut_ptr(), a.len()) });
            let mut output = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    mlx_sys::mlx_kquant_sg8_prepared(
                        a[0],
                        tiled.w.as_raw_ptr(),
                        tiled.scales.as_raw_ptr(),
                        tiled.biases.as_raw_ptr(),
                        a[1],
                        a[2],
                        &mut output,
                    )
                },
                1
            );
            let actual = read_bits("attention prepared projection", output);
            let reference = unsafe {
                mlx_sys::mlx_test_kquant_quantized_matmul(
                    a[0],
                    tiled.w.as_raw_ptr(),
                    tiled.scales.as_raw_ptr(),
                    tiled.biases.as_raw_ptr(),
                    true,
                    32,
                    4,
                    mode.as_ptr(),
                    GPU,
                )
            };
            assert_eq!(
                actual,
                read_bits("attention fresh packing projection", reference),
                "split={split}, seed={seed}"
            );
            for handle in a {
                unsafe { mlx_sys::mlx_array_delete(handle) };
            }
        }
    }
}

struct StorageEnv(&'static str, Option<std::ffi::OsString>);
impl StorageEnv {
    fn save(key: &'static str) -> Self {
        Self(key, std::env::var_os(key))
    }
}
impl Drop for StorageEnv {
    fn drop(&mut self) {
        unsafe {
            match &self.1 {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
        }
    }
}
fn storage_bits(a: Vec<*mut mlx_sys::mlx_array>) -> Vec<Vec<u32>> {
    a.into_iter()
        .enumerate()
        .map(|(i, h)| {
            if i == 8 {
                let len = unsafe { mlx_sys::mlx_array_size(h) };
                packed_bits(h, len)
            } else {
                read_bits("GDN storage result", h).2
            }
        })
        .collect()
}
unsafe extern "C" fn build_storage(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    count: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if count != 10 || ![8, 10].contains(&n_outputs) || context.is_null() {
        return false;
    }
    unsafe { &*context.cast::<std::sync::atomic::AtomicUsize>() }
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, count) }
        .iter()
        .map(|a| a.cast_mut())
        .collect();
    let launch = if n_outputs == 10 {
        mlx_sys::mlx_gdn_fused_complete_prepared
    } else {
        mlx_sys::mlx_gdn_fused_complete
    };
    let result = unsafe {
        launch(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            1e-6,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            outputs,
        )
    };
    for handle in a {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    result
}
#[test]
fn gdn_t8_storage_preserves_both_schedules_prepared_tables_and_replay() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    let _scan = StorageEnv::save("MLX_GDN_SCAN8");
    let _storage = StorageEnv::save("MLX_GDN_T8_STORAGE");
    for (schedule, scan) in ["0", "1"].into_iter().enumerate() {
        unsafe { std::env::set_var("MLX_GDN_SCAN8", scan) };
        for prepared in [false, true] {
            let key = 0x6DD8_0000_0000_0200 + schedule as u64 * 2 + u64::from(prepared);
            let mut traces = std::sync::atomic::AtomicUsize::new(0);
            let cases = if prepared {
                vec![(8, 32), (8, 48), (8, 32)]
            } else {
                vec![(8, 32), (7, 48), (1, 32), (16, 32), (8, 48)]
            };
            for (index, (rows, heads)) in cases.into_iter().enumerate() {
                let input = Inputs::new(rows, heads, index % 2 == 0, 331 + index as u32 * 17);
                unsafe { std::env::set_var("MLX_GDN_T8_STORAGE", "0") };
                let reference = storage_bits(input.call(false, prepared));
                unsafe { std::env::set_var("MLX_GDN_T8_STORAGE", "1") };
                let mut alias = input.call(true, prepared);
                assert!(unsafe { mlx_sys::mlx_eval(alias.as_mut_ptr(), alias.len()) });
                assert!(unsafe {
                    mlx_sys::mlx_array_aliases(alias[1], input.state_dst.as_raw_ptr())
                });
                assert!(unsafe {
                    mlx_sys::mlx_array_aliases(alias[7], input.history_dst.as_raw_ptr())
                });
                assert_eq!(
                    storage_bits(alias),
                    reference,
                    "eager scan={scan}, prepared={prepared}, rows={rows}"
                );
                let arrays: Vec<_> = input
                    .arrays
                    .iter()
                    .map(|a| a.as_raw_ptr().cast_const())
                    .collect();
                let mut output = vec![std::ptr::null_mut(); if prepared { 10 } else { 8 }];
                assert!(unsafe {
                    mlx_sys::mlx_compiled_graph_invoke(
                        key,
                        Some(build_storage),
                        (&mut traces as *mut std::sync::atomic::AtomicUsize).cast(),
                        arrays.as_ptr(),
                        arrays.len(),
                        output.as_mut_ptr(),
                        output.len(),
                        true,
                    )
                });
                start_counting();
                assert_eq!(
                    storage_bits(output),
                    reference,
                    "compiled scan={scan}, prepared={prepared}, rows={rows}"
                );
                assert_eq!(family_count("gdn_t8_storage"), u64::from(rows == 8));
                stop_counting();
            }
            assert_eq!(traces.load(std::sync::atomic::Ordering::Relaxed), 1);
            unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        }
    }
}
#[test]
#[ignore = "isolated GDN storage timing, coordinate the GPU slot"]
fn gdn_t8_storage_microbenchmark() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    let prepared = std::env::var("MLX_GDN_PREPARED_SG8").is_ok_and(|v| v == "1");
    let input = Inputs::new(8, 32, true, 431);
    let mut inputs: Vec<_> = input.arrays.iter().map(MxArray::as_raw_ptr).collect();
    assert!(unsafe { mlx_sys::mlx_eval(inputs.as_mut_ptr(), inputs.len()) });
    let mut samples = Vec::new();
    for sample in 0..57 {
        let mut roots: Vec<_> = (0..32).flat_map(|_| input.call(false, prepared)).collect();
        start_counting();
        let at = std::time::Instant::now();
        assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
        let elapsed = at.elapsed().as_secs_f64() * 1e6 / 32.0;
        assert_eq!(
            family_count("gdn_fused_complete"),
            32,
            "every timed batch must execute 32 independent kernels"
        );
        stop_counting();
        if sample >= 16 {
            samples.push(elapsed);
        }
        for h in roots {
            unsafe { mlx_sys::mlx_array_delete(h) };
        }
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "gdn_t8_storage={} scan8={} prepared={prepared} median_us={} p10_us={}",
        std::env::var("MLX_GDN_T8_STORAGE").unwrap_or_default(),
        std::env::var("MLX_GDN_SCAN8").unwrap_or_default(),
        samples[20],
        samples[4]
    );
}

#[test]
#[ignore = "serialized GDN per-layer latency, coordinate the GPU slot"]
fn gdn_t8_storage_serial_microbenchmark() {
    let _guard = SERIAL.lock().unwrap();
    if gpu_gen() < 17 {
        return;
    }
    let prepared = std::env::var("MLX_GDN_PREPARED_SG8").is_ok_and(|v| v == "1");
    let fixtures: Vec<_> = (0..32)
        .map(|i| Inputs::new(8, 32, true, 541 + i * 13))
        .collect();
    let mut inputs: Vec<_> = fixtures
        .iter()
        .flat_map(|f| f.arrays.iter().map(MxArray::as_raw_ptr))
        .collect();
    assert!(unsafe { mlx_sys::mlx_eval(inputs.as_mut_ptr(), inputs.len()) });
    let launch = if prepared {
        mlx_sys::mlx_gdn_fused_complete_prepared
    } else {
        mlx_sys::mlx_gdn_fused_complete
    };
    let mut samples = Vec::new();
    for sample in 0..57 {
        let mut roots = Vec::new();
        let mut state = fixtures[0].arrays[7].as_raw_ptr();
        let mut history = fixtures[0].arrays[4].as_raw_ptr();
        for fixture in &fixtures {
            let mut a: Vec<_> = fixture.arrays.iter().map(MxArray::as_raw_ptr).collect();
            a[7] = state;
            a[4] = history;
            let mut outputs = vec![std::ptr::null_mut(); if prepared { 10 } else { 8 }];
            assert!(unsafe {
                launch(
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    a[4],
                    a[5],
                    a[6],
                    a[7],
                    a[8],
                    a[9],
                    1e-6,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    outputs.as_mut_ptr(),
                )
            });
            state = outputs[1];
            history = outputs[7];
            roots.extend(outputs);
        }
        start_counting();
        let at = std::time::Instant::now();
        assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
        let elapsed = at.elapsed().as_secs_f64() * 1e6 / 32.0;
        assert_eq!(
            family_count("gdn_fused_complete"),
            32,
            "serialized kernels must not be deduplicated"
        );
        stop_counting();
        if sample >= 16 {
            samples.push(elapsed);
        }
        // Each kernel depends on the prior state and history; this measures
        // a serialized 32-kernel chain, not independent-grid throughput.
        for handle in roots {
            unsafe { mlx_sys::mlx_array_delete(handle) };
        }
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "gdn_t8_storage_serial={} scan8={} prepared={prepared} median_us={} p10_us={}",
        std::env::var("MLX_GDN_T8_STORAGE").unwrap_or_default(),
        std::env::var("MLX_GDN_SCAN8").unwrap_or_default(),
        samples[20],
        samples[4]
    );
}
