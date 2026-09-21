//! Generic `mlx::core::compile` bridge.
//!
//! [`invoke_compiled_graph`] registers a Rust graph builder under a stable
//! `fn_id` and calls its compiled form. The first invoke for an id traces the
//! builder on tracer arrays (fusion + tape recording); later invokes on the
//! same thread replay the fused tape with fresh inputs and never run Rust
//! code again — per-cycle graph construction collapses to one FFI call plus
//! `compile_replace` in C++.
//!
//! Builder contract (traced once per `fn_id`/shape-signature):
//! - MUST be a pure function of `inputs` — no `eval`/`item`/`to_*` reads, no
//!   mutation of outside state, no host-integer reads that vary per call
//!   (pass such values as array inputs instead).
//! - MUST return exactly `n_outputs` arrays.
//! - Captured non-input arrays (weights) become tape constants: materialize
//!   them BEFORE the first invoke or their producer chains bake into the
//!   tape and recompute every call.
//!
//! With `shapeless = true`, input *shapes* leave the cache key — only valid
//! when no traced node bakes a shape-derived host int (see `mlx/compile.cpp`
//! `CompilerCache::find`). `MLX_DISABLE_COMPILE` (MLX built-in) and any
//! caller-side kill-switch keep an eager fallback path.

use std::ffi::c_void;

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

type BuilderFn<'a> = dyn FnMut(&[MxArray]) -> Result<Vec<MxArray>> + 'a;

struct BuilderCtx<'a> {
    builder: &'a mut BuilderFn<'a>,
    err: Option<String>,
}

unsafe extern "C" fn builder_trampoline(
    ctx: *mut c_void,
    inputs: *const *const sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut sys::mlx_array,
    n_outputs: usize,
) -> bool {
    let ctx = unsafe { &mut *(ctx as *mut BuilderCtx<'_>) };
    // Wrap each traced input: the C++ side allocated owning copies for the
    // builder, so `from_handle` ownership is correct. Null input = FFI bug.
    let mut wrapped = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let ptr = unsafe { *inputs.add(i) };
        if ptr.is_null() {
            ctx.err = Some("compiled graph builder received a null input".into());
            return false;
        }
        // const-cast is safe: the copies are exclusively owned by this call.
        match MxArray::from_handle(ptr as *mut sys::mlx_array, "compiled graph input") {
            Ok(a) => wrapped.push(a),
            Err(e) => {
                ctx.err = Some(format!("compiled graph input wrap failed: {e}"));
                return false;
            }
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.builder)(&wrapped)));
    let out = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            ctx.err = Some(format!("{e}"));
            return false;
        }
        Err(_) => {
            ctx.err = Some("compiled graph builder panicked".into());
            return false;
        }
    };
    if out.len() != n_outputs {
        ctx.err = Some(format!(
            "compiled graph builder produced {} outputs, expected {n_outputs}",
            out.len()
        ));
        return false;
    }
    for (i, arr) in out.iter().enumerate() {
        // Owning handle copy: `mlx_array_clone_handle` bumps the shared
        // ArrayDesc refcount without adding a graph node, so the tape keeps
        // the array alive after `out` drops the Rust-side handles.
        let ptr = unsafe { sys::mlx_array_clone_handle(arr.handle.0) };
        if ptr.is_null() {
            ctx.err = Some("compiled graph output clone failed".into());
            return false;
        }
        unsafe { *outputs.add(i) = ptr };
    }
    true
}

/// Invoke the compiled graph registered under `fn_id`, tracing `builder` on
/// first call. `inputs` are borrowed; `builder` runs at most once per
/// (fn_id, dtype/shape-signature) on this thread.
///
/// Returns `None` when the FFI reports failure (no Metal device, builder
/// error, arity mismatch) — callers fall back to their eager path. Builder
/// errors are returned as `Err` so genuine graph bugs surface.
pub(crate) fn invoke_compiled_graph(
    fn_id: u64,
    inputs: &[&MxArray],
    n_outputs: usize,
    shapeless: bool,
    builder: &mut BuilderFn<'_>,
) -> Result<Option<Vec<MxArray>>> {
    let input_ptrs: Vec<*const sys::mlx_array> =
        inputs.iter().map(|a| a.as_raw_ptr() as *const _).collect();
    let mut output_ptrs: Vec<*mut sys::mlx_array> = vec![std::ptr::null_mut(); n_outputs];
    let mut ctx = BuilderCtx { builder, err: None };
    let ok = unsafe {
        sys::mlx_compiled_graph_invoke(
            fn_id,
            Some(builder_trampoline),
            &mut ctx as *mut BuilderCtx<'_> as *mut c_void,
            input_ptrs.as_ptr(),
            input_ptrs.len(),
            output_ptrs.as_mut_ptr(),
            output_ptrs.len(),
            shapeless,
        )
    };
    if !ok {
        return match ctx.err {
            Some(msg) => Err(Error::from_reason(format!(
                "compiled graph {fn_id:#x} builder failed: {msg}"
            ))),
            None => Ok(None),
        };
    }
    let mut out = Vec::with_capacity(n_outputs);
    for ptr in output_ptrs {
        if ptr.is_null() {
            return Err(Error::from_reason(format!(
                "compiled graph {fn_id:#x} returned a null output"
            )));
        }
        out.push(MxArray::from_handle(ptr, "compiled graph output")?);
    }
    Ok(Some(out))
}

/// Erase every cached compiled-graph entry whose fn_id matches `value` under
/// `mask`. Compiled tapes retain their captured constants (model weights), so
/// models must erase their ids on drop or a reload keeps stale weights
/// resident for the process lifetime.
pub(crate) fn erase_compiled_graphs_matching(mask: u64, value: u64) {
    unsafe { sys::mlx_compiled_graph_erase_matching(mask, value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn fn_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Dedicated high bit so test ids can never collide with a heap
        // pointer `compile` derives for C++-side call sites.
        0xDFC0_0000_0000_0000 | NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// The compile mode is process-global, and the disable test flips it —
    /// serialize every test that goes through `invoke_compiled_graph`.
    static COMPILE_MODE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn compiled_graph_traces_once_and_replays() {
        let _guard = COMPILE_MODE_LOCK.lock().unwrap();
        let id = fn_id();
        let calls = std::cell::Cell::new(0usize);
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            calls.set(calls.get() + 1);
            Ok(vec![inputs[0].add(&inputs[1])?])
        };
        let mut run = |a: &[f32], b: &[f32], shape: &[i64]| -> Vec<f32> {
            eprintln!("[cg-test] invoke start");
            let a = MxArray::from_float32(a, shape).unwrap();
            let b = MxArray::from_float32(b, shape).unwrap();
            let out = invoke_compiled_graph(id, &[&a, &b], 1, false, &mut builder)
                .unwrap()
                .expect("compiled invoke");
            eprintln!("[cg-test] invoke returned, eval");
            out[0].eval();
            eprintln!("[cg-test] eval done");
            out[0].to_float32().unwrap().to_vec()
        };
        assert_eq!(run(&[1.0, 2.0], &[10.0, 20.0], &[2]), [11.0, 22.0]);
        assert_eq!(run(&[3.0, 4.0], &[30.0, 40.0], &[2]), [33.0, 44.0]);
        assert_eq!(calls.get(), 1, "cache hit must not re-run the builder");
    }

    #[test]
    fn compiled_graph_shapeless_replays_across_shapes() {
        let _guard = COMPILE_MODE_LOCK.lock().unwrap();
        let id = fn_id();
        // add(mul(a,b),a) is a fusible elementwise chain — the tape carries a
        // Compiled node, so the shapeless hit path exercises
        // Compiled::output_shapes through compile_replace.
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            Ok(vec![inputs[0].mul(&inputs[1])?.add(&inputs[0])?])
        };
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap();
        let out = invoke_compiled_graph(id, &[&a, &b], 1, true, &mut builder)
            .unwrap()
            .expect("compiled invoke");
        out[0].eval();
        // add(mul(a,b),a) = a*b + a
        assert_eq!(out[0].to_float32().unwrap().as_ref(), [3.0, 6.0, 9.0]);
        // Different shape, same dtype/rank: shapeless hits the same entry.
        let a = MxArray::from_float32(&[5.0, 6.0], &[2]).unwrap();
        let b = MxArray::from_float32(&[3.0, 3.0], &[2]).unwrap();
        let out = invoke_compiled_graph(id, &[&a, &b], 1, true, &mut builder)
            .unwrap()
            .expect("shapeless hit");
        out[0].eval();
        assert_eq!(out[0].to_float32().unwrap().as_ref(), [20.0, 24.0]);
    }

    #[test]
    fn compiled_graph_builder_error_propagates() {
        let _guard = COMPILE_MODE_LOCK.lock().unwrap();
        let id = fn_id();
        let mut builder = |_inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            Err(Error::from_reason("intentional builder failure"))
        };
        let a = MxArray::from_float32(&[1.0], &[1]).unwrap();
        let result = invoke_compiled_graph(id, &[&a], 1, false, &mut builder);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("builder failure must surface as Err"),
        };
        assert!(err.to_string().contains("intentional builder failure"));
    }

    #[test]
    fn compiled_graph_retrace_uses_fresh_builder_ctx() {
        let _guard = COMPILE_MODE_LOCK.lock().unwrap();
        let id = fn_id();
        let calls = std::cell::Cell::new(0usize);
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            calls.set(calls.get() + 1);
            Ok(vec![inputs[0].add(&inputs[1])?])
        };
        let mut run = |a: &[f32], b: &[f32], shape: &[i64]| -> Vec<f32> {
            let a = MxArray::from_float32(a, shape).unwrap();
            let b = MxArray::from_float32(b, shape).unwrap();
            let out = invoke_compiled_graph(id, &[&a, &b], 1, false, &mut builder)
                .unwrap()
                .expect("compiled invoke");
            out[0].eval();
            out[0].to_float32().unwrap().to_vec()
        };
        assert_eq!(run(&[1.0, 2.0], &[10.0, 20.0], &[2]), [11.0, 22.0]);
        // A new shape signature under shapeless=false re-traces: the cached
        // closure runs the builder again. The ctx it sees must come from the
        // per-invoke slot — not the first call's (now-dead) stack frame.
        assert_eq!(
            run(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[3]),
            [5.0, 7.0, 9.0]
        );
        assert_eq!(calls.get(), 2, "new shape signature must re-trace");
        // And the cached tape for the first signature still replays.
        assert_eq!(run(&[7.0, 8.0], &[1.0, 1.0], &[2]), [8.0, 9.0]);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn compiled_graph_disabled_mode_reruns_builder_with_fresh_ctx() {
        let _guard = COMPILE_MODE_LOCK.lock().unwrap();
        unsafe { sys::mlx_compiled_graph_set_compile_disabled(true) };
        let reset = scopeguard_reset();
        let id = fn_id();
        let calls = std::cell::Cell::new(0usize);
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            calls.set(calls.get() + 1);
            Ok(vec![inputs[0].add(&inputs[1])?])
        };
        let mut run = |a: &[f32], b: &[f32]| -> Vec<f32> {
            let a = MxArray::from_float32(a, &[2]).unwrap();
            let b = MxArray::from_float32(b, &[2]).unwrap();
            let out = invoke_compiled_graph(id, &[&a, &b], 1, false, &mut builder)
                .unwrap()
                .expect("invoke");
            out[0].eval();
            out[0].to_float32().unwrap().to_vec()
        };
        // With compile disabled, `compile()` returns the raw closure which
        // runs on EVERY invoke — each with a different Rust-side ctx stack
        // frame. Every call must see its own live ctx.
        assert_eq!(run(&[1.0, 2.0], &[10.0, 20.0]), [11.0, 22.0]);
        assert_eq!(run(&[3.0, 4.0], &[30.0, 40.0]), [33.0, 44.0]);
        assert_eq!(run(&[5.0, 6.0], &[50.0, 60.0]), [55.0, 66.0]);
        assert_eq!(
            calls.get(),
            3,
            "disabled mode must run the builder per call"
        );
        drop(reset);
    }

    fn scopeguard_reset() -> impl Drop {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                unsafe { sys::mlx_compiled_graph_set_compile_disabled(false) };
            }
        }
        Reset
    }
}
