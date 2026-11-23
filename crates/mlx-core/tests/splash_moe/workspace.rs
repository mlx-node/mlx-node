//! Exact allocation-only A/B on actual eligible weights; no second model.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
struct Flags(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl Flags {
    fn new(keys: &[&'static str]) -> Self {
        Self(
            keys.iter()
                .map(|&key| (key, std::env::var_os(key)))
                .collect(),
        )
    }
}
impl Drop for Flags {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
fn flag(name: &str, enabled: bool) {
    unsafe { std::env::set_var(name, if enabled { "1" } else { "0" }) }
}
struct Owned(*mut mlx_sys::mlx_array);
impl Owned {
    fn new(h: *mut mlx_sys::mlx_array) -> Self {
        assert!(!h.is_null());
        Self(h)
    }
    fn bits(&self) -> Vec<u32> {
        let mut h = self.0;
        let mut error = [0i8; 1024];
        assert!(
            unsafe { mlx_sys::mlx_eval_with_error(&mut h, 1, error.as_mut_ptr(), error.len()) },
            "{error:?}"
        );
        let len = unsafe { mlx_sys::mlx_array_size(h) };
        if unsafe { mlx_sys::mlx_array_dtype(h) } == DType::Uint32 as i32 {
            let mut v = vec![0u32; len];
            assert!(unsafe { mlx_sys::mlx_array_to_uint32(h, v.as_mut_ptr(), len) });
            v
        } else {
            assert_eq!(
                unsafe { mlx_sys::mlx_array_dtype(h) },
                DType::BFloat16 as i32
            );
            let mut v = vec![0u16; len];
            assert!(unsafe { mlx_sys::mlx_array_to_uint16(h, v.as_mut_ptr(), len) });
            v.into_iter().map(u32::from).collect()
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_array_delete(self.0) }
    }
}
fn raw(a: &[*mut mlx_sys::mlx_array], modes: &[u32]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_splash_moe(a.as_ptr(), a.len(), modes.as_ptr(), 8, true) }
}
fn eager(a: &[MxArray], modes: &[u32]) -> Owned {
    let a: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
    Owned::new(raw(&a, modes))
}
struct Builder {
    modes: Vec<u32>,
    traces: AtomicUsize,
}
struct Inputs(Vec<*mut mlx_sys::mlx_array>);
impl Drop for Inputs {
    fn drop(&mut self) {
        for h in self.0.drain(..) {
            unsafe { mlx_sys::mlx_array_delete(h) }
        }
    }
}
unsafe extern "C" fn build(
    ctx: *mut std::ffi::c_void,
    input: *const *const mlx_sys::mlx_array,
    n: usize,
    out: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if input.is_null() {
        return false;
    }
    let a = Inputs(
        unsafe { std::slice::from_raw_parts(input, n) }
            .iter()
            .map(|h| h.cast_mut())
            .collect(),
    );
    if ctx.is_null() || n != 22 || m != 1 {
        return false;
    }
    let b = unsafe { &*ctx.cast::<Builder>() };
    b.traces.fetch_add(1, Ordering::Relaxed);
    let h = raw(&a.0, &b.modes);
    unsafe { *out = h };
    !h.is_null()
}
unsafe extern "C" fn build_chain(
    ctx: *mut std::ffi::c_void,
    input: *const *const mlx_sys::mlx_array,
    n: usize,
    out: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if input.is_null() {
        return false;
    }
    let a = Inputs(
        unsafe { std::slice::from_raw_parts(input, n) }
            .iter()
            .map(|h| h.cast_mut())
            .collect(),
    );
    if ctx.is_null() || n != 22 || m != 2 {
        return false;
    }
    let b = unsafe { &*ctx.cast::<Builder>() };
    b.traces.fetch_add(1, Ordering::Relaxed);
    let first = raw(&a.0, &b.modes);
    if first.is_null() {
        return false;
    }
    let mut next = a.0.clone();
    next[0] = first;
    let second = raw(&next, &b.modes);
    if second.is_null() {
        unsafe { mlx_sys::mlx_array_delete(first) };
        return false;
    }
    unsafe {
        *out = first;
        *out.add(1) = second
    };
    true
}
fn chain_replay(saved: &[MxArray], modes: &[u32]) {
    let key = 0x57534b434841494e;
    let _erase = Erase(key);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut b = Builder {
        modes: modes.to_vec(),
        traces: AtomicUsize::new(0),
    };
    for i in 0..3 {
        let mut a = saved.to_vec();
        a[0] = a[0]
            .mul_scalar(0.875 + i as f64 * 0.0625)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        a.push(MxArray::from_int32(&[8], &[]).unwrap());
        flag("MLX_SPLASH_MOE_WORKSPACE", false);
        let first = eager(&a, modes);
        let mut next: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
        next[0] = first.0;
        let second = Owned::new(raw(&next, modes));
        let expected = [first.bits(), second.bits()];
        flag("MLX_SPLASH_MOE_WORKSPACE", true);
        let h: Vec<_> = a.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
        let mut outputs = [std::ptr::null_mut(); 2];
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build_chain),
                (&mut b as *mut Builder).cast(),
                h.as_ptr(),
                h.len(),
                outputs.as_mut_ptr(),
                2,
                true,
            )
        });
        let owned = [Owned::new(outputs[0]), Owned::new(outputs[1])];
        start_counting();
        assert!(unsafe { mlx_sys::mlx_eval(outputs.as_mut_ptr(), 2) });
        let selections = family_count("splash_moe_workspace");
        stop_counting();
        assert_eq!(selections, 2, "dependent chain must select pool twice");
        assert_eq!([owned[0].bits(), owned[1].bits()], expected);
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        eprintln!(
            "WORKSPACE_DEPENDENT_CHAIN replay={i} pool_selections={selections} outputs=2 exact"
        );
    }
}
struct Erase(u64);
impl Drop for Erase {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) }
    }
}
fn rows(a: &MxArray, m: i64) -> MxArray {
    let copies = vec![a; ((m + 7) / 8) as usize];
    MxArray::concatenate_many(copies, Some(0))
        .unwrap()
        .slice(&[0, 0], &[m, a.shape().unwrap()[1]])
        .unwrap()
}
pub(super) fn actual_workspace_replay(saved: &[MxArray], modes: &[u32]) {
    assert_eq!(saved.len(), 21);
    assert_eq!(saved[0].shape().unwrap().as_ref(), [8, 2048]);
    let _flags = Flags::new(&["MLX_SPLASH_MOE_WORKSPACE", "MLX_SPLASH_MOE_REGISTER"]);
    flag("MLX_SPLASH_MOE_REGISTER", false);
    let mut held: Vec<(Owned, Vec<u32>)> = Vec::new();
    let key = 0x57534b5245504c00u64;
    let _erase = Erase(key);
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut b = Builder {
        modes: modes.to_vec(),
        traces: AtomicUsize::new(0),
    };
    for (iteration, (m, budget)) in [
        (8i64, 8i32),
        (1, 1),
        (7, 3),
        (33, 8),
        (8, 1),
        (8, 3),
        (8, 8),
    ]
    .into_iter()
    .enumerate()
    {
        let mut a = saved.to_vec();
        for value in a.iter_mut().take(3) {
            *value = rows(value, m)
                .mul_scalar(1.0 + iteration as f64 * 0.015625)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
        }
        // Force a small byte offset on the final input replay to retain the
        // independent normalization-copy path alongside persistent slots.
        if iteration == 6 {
            let shape = a[0].shape().unwrap();
            let size = shape.iter().product::<i64>();
            let flat = a[0].reshape(&[-1]).unwrap();
            let pad = MxArray::zeros(&[1], Some(DType::BFloat16)).unwrap();
            let storage = MxArray::concatenate_many(vec![&pad, &flat], Some(0)).unwrap();
            storage.eval();
            a[0] = storage
                .slice(&[1], &[size + 1])
                .unwrap()
                .reshape(&shape)
                .unwrap();
            a[0].eval();
            assert_ne!(
                unsafe { mlx_sys::mlx_array_get_buffer_offset(a[0].as_raw_ptr()) } % 16,
                0
            );
        }
        a.push(MxArray::from_int32(&[budget], &[]).unwrap());
        flag("MLX_SPLASH_MOE_WORKSPACE", false);
        let expected = eager(&a, modes).bits();
        flag("MLX_SPLASH_MOE_WORKSPACE", true);
        start_counting();
        let direct = eager(&a, modes);
        assert_eq!(direct.bits(), expected);
        assert_eq!(family_count("splash_moe_workspace"), u64::from(m == 8));
        stop_counting();
        // Only trace-time true may select the pool: later process flips
        // must not mutate this graph's captured allocation policy.
        flag("MLX_SPLASH_MOE_WORKSPACE", iteration == 0);
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
        start_counting();
        let replay = Owned::new(out);
        assert_eq!(replay.bits(), expected);
        assert_eq!(family_count("splash_moe_workspace"), u64::from(m == 8));
        stop_counting();
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        held.push((replay, expected));
        for (prior, expected) in &held {
            assert_eq!(
                &prior.bits(),
                expected,
                "old public output aliases overwritten scratch"
            );
        }
        eprintln!(
            "WORKSPACE_EXACT rows={m} budget={budget} replay={iteration} held={}",
            held.len()
        );
    }
    chain_replay(saved, modes);
    // A profile publishes a scratch scalar. It must stay fresh and stable even
    // when production calls reuse the workspace afterward. Stage probes also
    // retain independent allocation; neither may record a workspace selection.
    flag("MLX_SPLASH_MOE_WORKSPACE", true);
    let h: Vec<_> = saved.iter().map(MxArray::as_raw_ptr).collect();
    let mut diagnostics = Vec::new();
    for stage in 1..=6 {
        start_counting();
        let out = Owned::new(unsafe {
            mlx_sys::mlx_test_splash_moe_profile(
                h.as_ptr(),
                h.len(),
                modes.as_ptr(),
                8,
                true,
                stage,
            )
        });
        let bits = out.bits();
        assert_eq!(family_count("splash_moe_workspace"), 0);
        stop_counting();
        diagnostics.push((out, bits));
    }
    start_counting();
    let probe = Owned::new(unsafe {
        mlx_sys::mlx_test_splash_moe_stage(h.as_ptr(), h.len(), modes.as_ptr(), 8, true, 4)
    });
    let probe_bits = probe.bits();
    assert_eq!(family_count("splash_moe_workspace"), 0);
    stop_counting();
    let _ = eager(saved, modes).bits();
    for (out, bits) in diagnostics {
        assert_eq!(out.bits(), bits);
    }
    assert_eq!(probe.bits(), probe_bits);
    flag("MLX_SPLASH_MOE_REGISTER", true);
    start_counting();
    let _ = eager(saved, modes).bits();
    assert_eq!(family_count("splash_moe_workspace"), 0);
    assert_eq!(family_count("splash_moe_register_gate_up"), 1);
    stop_counting();
}
#[test]
#[ignore = "exclusive GPU, actual checkpoint workspace exactness and one-trace replay"]
fn grouped_workspace_actual_reuse_keeps_old_outputs_and_shape_fallbacks() {
    let _flags = Flags::new(&["SPLASH_MOE_ROWS", "SPLASH_MOE_WORKSPACE_CHECK"]);
    unsafe {
        std::env::set_var("SPLASH_MOE_ROWS", "8");
        std::env::set_var("SPLASH_MOE_WORKSPACE_CHECK", "1");
    }
    grouped_actual_layer_stage_comparison();
}
