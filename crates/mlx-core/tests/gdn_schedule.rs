//! Isolate Splash's eight-SIMD-group/two-state-row scheduling from the
//! established native GDN arithmetic and cache ownership contracts.
mod kquant_support;
use kquant_support::{activation, read_bits};
use mlx_core::array::{DType, MxArray};
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Compiled(u64);
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}

unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || n_inputs != 10 || n_outputs != 8 {
        return false;
    }
    let traces = unsafe { &*context.cast::<std::sync::atomic::AtomicUsize>() };
    traces.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n_inputs) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let result = unsafe {
        mlx_sys::mlx_gdn_fused_complete(
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
    // The graph callback owns each traced input handle. The primitive has
    // retained its array values, so release the temporary handles on either result.
    for handle in a {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    result
}

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
    fn call(&self, destinations: bool) -> Vec<*mut mlx_sys::mlx_array> {
        let a: Vec<_> = self.arrays.iter().map(MxArray::as_raw_ptr).collect();
        let mut out = vec![std::ptr::null_mut(); 8];
        assert!(unsafe {
            mlx_sys::mlx_gdn_fused_complete(
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

#[test]
fn gdn_scan8_preserves_all_outputs_and_destination_aliases() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    for (rows, heads, strided, seed) in [
        (8, 32, false, 17),
        (8, 32, true, 29),
        (8, 48, true, 41),
        (1, 32, true, 53),
        (7, 32, false, 67),
        (16, 32, true, 79),
    ] {
        let input = Inputs::new(rows, heads, strided, seed);
        unsafe {
            std::env::set_var("MLX_GDN_SCAN8", "0");
        }
        let reference: Vec<_> = input
            .call(false)
            .into_iter()
            .map(|h| read_bits("GDN reference", h))
            .collect();
        unsafe {
            std::env::set_var("MLX_GDN_SCAN8", "1");
        }
        let mut actual = input.call(true);
        assert!(unsafe { mlx_sys::mlx_eval(actual.as_mut_ptr(), actual.len()) });
        assert!(unsafe { mlx_sys::mlx_array_aliases(actual[1], input.state_dst.as_raw_ptr()) });
        assert!(unsafe { mlx_sys::mlx_array_aliases(actual[7], input.history_dst.as_raw_ptr()) });
        for (slot, (got, want)) in actual
            .into_iter()
            .map(|h| read_bits("GDN scan8", h))
            .zip(reference)
            .enumerate()
        {
            assert_eq!(
                got, want,
                "rows={rows} heads={heads} strided={strided} slot={slot}"
            );
        }
    }
    unsafe {
        std::env::remove_var("MLX_GDN_SCAN8");
    }
}

#[test]
fn gdn_scan8_compiled_replay_selects_schedule_from_current_width() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let compiled = Compiled(0x6DD8_0000_0000_0001);
    let mut traces = std::sync::atomic::AtomicUsize::new(0);
    for (index, rows) in [8, 7, 1, 16, 8].into_iter().enumerate() {
        let input = Inputs::new(rows, 32, true, 101 + index as u32);
        unsafe { std::env::set_var("MLX_GDN_SCAN8", "0") };
        let reference: Vec<_> = input
            .call(false)
            .into_iter()
            .map(|h| read_bits("GDN replay reference", h))
            .collect();
        unsafe { std::env::set_var("MLX_GDN_SCAN8", "1") };
        let arrays: Vec<_> = input
            .arrays
            .iter()
            .map(|a| a.as_raw_ptr().cast_const())
            .collect();
        let mut outputs = vec![std::ptr::null_mut(); 8];
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                compiled.0,
                Some(build),
                (&mut traces as *mut std::sync::atomic::AtomicUsize).cast(),
                arrays.as_ptr(),
                arrays.len(),
                outputs.as_mut_ptr(),
                outputs.len(),
                true,
            )
        });
        for (slot, (got, want)) in outputs
            .into_iter()
            .map(|h| read_bits("GDN replay", h))
            .zip(reference)
            .enumerate()
        {
            assert_eq!(got, want, "replay rows={rows} slot={slot}");
        }
        assert_eq!(
            traces.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "row count must change the schedule without retracing"
        );
    }
    unsafe { std::env::remove_var("MLX_GDN_SCAN8") };
}

#[test]
#[ignore = "isolated operator timing, coordinate the GPU slot"]
fn gdn_scan8_microbenchmark() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let input = Inputs::new(8, 32, true, 83);
    let mut inputs: Vec<_> = input.arrays.iter().map(MxArray::as_raw_ptr).collect();
    assert!(unsafe { mlx_sys::mlx_eval(inputs.as_mut_ptr(), inputs.len()) });
    let mut samples = Vec::new();
    for i in 0..57 {
        let mut roots: Vec<_> = (0..32).flat_map(|_| input.call(false)).collect();
        let at = std::time::Instant::now();
        assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
        let us = at.elapsed().as_secs_f64() * 1e6 / 32.0;
        if i >= 16 {
            samples.push(us);
        }
        for h in roots {
            unsafe {
                mlx_sys::mlx_array_delete(h);
            }
        }
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "gdn_scan8={} median_us={} p10_us={}",
        std::env::var("MLX_GDN_SCAN8").unwrap_or_default(),
        samples[20],
        samples[4]
    );
}
