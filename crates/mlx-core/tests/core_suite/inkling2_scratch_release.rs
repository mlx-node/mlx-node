//! Regression tests for Inkling2 persistent Metal scratch release.
//!
//! Verifies that the three `thread_local` Inkling2 Metal scratch arenas:
//! - `try_inkling2_v2_scratch` (`mlx_paged_dispatch.cpp`, >= 1,081,344 B)
//! - `inkling2_oproj_scratch` (`mlx_inkling2_ops.cpp`, >= 147,456 B)
//! - `inkling2_loop_ctrl_scratch` (`mlx_inkling2_ops.cpp`, >= 958,464 B)
//!
//! 1. Are released automatically when a worker thread exits (zero monotonic
//!    `mlx_get_active_memory()` growth across repeated worker-thread cycles).
//! 2. Are released on the current thread by both `mlx_clear_cache()` and
//!    `mlx_clear_streams()`, and re-acquire cleanly on a fresh stream index.

use std::sync::Mutex;

use mlx_core::array::{DType, MxArray, clear_cache, synchronize, synchronize_and_clear_cache};
use mlx_core::models::inkling2::kernels::{
    Inkling2KvDtype, inkling2_loop_ctrl_fused_single, inkling2_oproj_sconv_k4_ragged_tree,
    inkling2_paged_attention,
};

/// Serialize active-memory assertions across tests in this binary because
/// `mlx_get_active_memory()` reports process-wide allocator state.
static ACTIVE_MEMORY_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Combined minimum size of all three persistent Inkling2 scratch arenas:
/// - V2 paged-attention scratch: `(4096 * 2 + 262_144) * 4 = 1_081_344` bytes
/// - `o_proj + sconv` scratch:   `32 * 2304 * 2 = 147_456` bytes
/// - `loop_ctrl` scratch:        `(3*32*64 + 3*32*2304 + 3*32*128) * 4 = 958_464` bytes
///
/// Total = `2_187_264` bytes. Any two arenas alone are at most `2_039_808` bytes,
/// so asserting `>= baseline + 2_100_000` proves all three scratch paths allocated.
const MIN_COMBINED_SCRATCH_BYTES: u64 = 2_100_000;

fn metal_available() -> bool {
    // SAFETY: nullary Metal capability query.
    unsafe { mlx_sys::mlx_metal_is_available() }
}

fn active_memory_bytes() -> u64 {
    let mut bytes = 0u64;
    // SAFETY: `bytes` is a valid out-pointer.
    let rc = unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) };
    assert_eq!(rc, 0, "mlx_get_active_memory failed with rc={rc}");
    bytes
}

/// Exercises all three live Inkling2 persistent Metal scratch arenas on the
/// current thread's default GPU stream, evaluates all outputs, and drops every
/// caller-owned `MxArray` before returning.
fn run_all_inkling2_scratch_paths() {
    // 1. Inkling2 V2 paged attention (`seq_len = 640 > 512` forces V2 scratch).
    {
        let num_blocks = 20i64;
        let block_size = 32i32;
        let num_q_heads = 3i32;
        let num_kv_heads = 1i32;
        let head_size = 128i32;
        let total_q = 2i64;

        let q = MxArray::zeros(
            &[total_q, i64::from(num_q_heads), i64::from(head_size)],
            Some(DType::BFloat16),
        )
        .expect("q");
        let k_pool = MxArray::zeros(
            &[
                num_blocks,
                i64::from(num_kv_heads),
                16,
                i64::from(block_size),
                8,
            ],
            Some(DType::BFloat16),
        )
        .expect("k_pool");
        let v_pool = MxArray::zeros(
            &[
                num_blocks,
                i64::from(num_kv_heads),
                i64::from(head_size),
                i64::from(block_size),
            ],
            Some(DType::BFloat16),
        )
        .expect("v_pool");
        let bt_vals: Vec<i32> = (0..num_blocks as i32).collect();
        let block_table = MxArray::from_int32(&bt_vals, &[1, num_blocks]).expect("block_table");
        let seq_lens = MxArray::from_int32(&[640], &[1]).expect("seq_lens");
        let cu_seqlens_q = MxArray::from_int32(&[0, total_q as i32], &[2]).expect("cu_seqlens_q");
        let k_scale = MxArray::from_float32(&[1.0], &[1]).expect("k_scale");
        let v_scale = MxArray::from_float32(&[1.0], &[1]).expect("v_scale");

        let (out, lse) = inkling2_paged_attention(
            &q,
            &k_pool,
            &v_pool,
            &block_table,
            &seq_lens,
            &cu_seqlens_q,
            &k_scale,
            &v_scale,
            None,
            1.0 / (head_size as f32).sqrt(),
            0,
            0,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_size,
            Inkling2KvDtype::Bf16,
            None,
        )
        .expect("inkling2_paged_attention V2");
        out.eval();
        lse.eval();
    }

    // 2. Fused `o_proj + sconv` (`emit_tape = false` hits `inkling2_oproj_scratch`).
    {
        let rows = 4i64;
        let in_dim = 64i64;
        let out_dim = 64i64;

        let y = MxArray::zeros(&[rows, in_dim], Some(DType::BFloat16)).expect("y");
        let weight = MxArray::zeros(&[out_dim, in_dim], Some(DType::BFloat16)).expect("weight");
        let conv_w = MxArray::zeros(&[out_dim, 4], Some(DType::BFloat16)).expect("conv_w");
        let hist = MxArray::zeros(&[3, out_dim], Some(DType::BFloat16)).expect("hist");

        let (out_y, out_hist, tape) = inkling2_oproj_sconv_k4_ragged_tree(
            &y,
            &weight,
            None,
            &conv_w,
            None,
            None,
            &[&hist],
            1,
            false,
            None,
            false,
        )
        .expect("inkling2_oproj_sconv_k4_ragged_tree");
        assert!(tape.is_none());
        out_y.eval();
        out_hist.eval();
    }

    // 3. Fused `loop_ctrl` (hits `inkling2_loop_ctrl_scratch`).
    {
        let rows = 4i64;
        let dim = 64i64;
        let rank = 16i64;

        let dest = MxArray::zeros(&[rows, dim], Some(DType::BFloat16)).expect("dest");
        let msg = MxArray::zeros(&[rows, dim], Some(DType::BFloat16)).expect("msg");
        let down_w = MxArray::zeros(&[rank, 2 * dim], Some(DType::Float32)).expect("down_w");
        let gain_forget_w =
            MxArray::zeros(&[2 * dim, rank], Some(DType::Float32)).expect("gain_forget_w");
        let norm_weight = MxArray::zeros(&[dim], Some(DType::Float32)).expect("norm_weight");

        let out = inkling2_loop_ctrl_fused_single(
            &dest,
            &msg,
            &down_w,
            &gain_forget_w,
            &norm_weight,
            1e-6,
            1e-6,
            DType::BFloat16,
        )
        .expect("inkling2_loop_ctrl_fused_single");
        out.eval();
    }
}

#[test]
fn worker_thread_cycles_release_inkling2_scratch() {
    if !metal_available() {
        return;
    }
    let _guard = ACTIVE_MEMORY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // One warm-up cycle initializes Metal pipelines and allocator state before
    // capturing the baseline active memory.
    std::thread::spawn(|| {
        run_all_inkling2_scratch_paths();
        synchronize();
    })
    .join()
    .expect("warmup worker panicked");
    synchronize_and_clear_cache();
    let baseline = active_memory_bytes();

    for cycle in 0..3 {
        let held = std::thread::spawn(|| {
            run_all_inkling2_scratch_paths();
            synchronize();
            active_memory_bytes()
        })
        .join()
        .expect("worker thread panicked");

        assert!(
            held >= baseline + MIN_COMBINED_SCRATCH_BYTES,
            "cycle {cycle}: expected all 3 persistent scratch arenas live before thread exit \
             (held={held}, baseline={baseline}, min_delta={MIN_COMBINED_SCRATCH_BYTES})"
        );

        synchronize_and_clear_cache();
        let after = active_memory_bytes();
        assert!(
            after <= baseline,
            "cycle {cycle}: active memory leaked after worker thread exit \
             (after={after}, baseline={baseline}, leaked={})",
            after.saturating_sub(baseline)
        );
    }
}

#[test]
fn same_thread_clear_cache_and_clear_streams_release_inkling2_scratch() {
    if !metal_available() {
        return;
    }
    let _guard = ACTIVE_MEMORY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // Warm up on this thread and establish baseline.
    run_all_inkling2_scratch_paths();
    synchronize_and_clear_cache();
    let baseline = active_memory_bytes();

    // Verify `clear_cache()` releases this thread's persistent scratch arenas.
    run_all_inkling2_scratch_paths();
    synchronize();
    let held_before_clear_cache = active_memory_bytes();
    assert!(
        held_before_clear_cache >= baseline + MIN_COMBINED_SCRATCH_BYTES,
        "expected persistent scratch live before clear_cache \
         (held={held_before_clear_cache}, baseline={baseline})"
    );
    clear_cache();
    let after_clear_cache = active_memory_bytes();
    assert!(
        after_clear_cache <= baseline,
        "mlx_clear_cache did not release Inkling2 scratch \
         (after={after_clear_cache}, baseline={baseline})"
    );

    // Verify re-acquisition after `clear_cache()` works cleanly on the same thread.
    run_all_inkling2_scratch_paths();
    synchronize();
    let held_after_reacquire = active_memory_bytes();
    assert!(
        held_after_reacquire >= baseline + MIN_COMBINED_SCRATCH_BYTES,
        "expected persistent scratch re-acquired after clear_cache \
         (held={held_after_reacquire}, baseline={baseline})"
    );
    synchronize_and_clear_cache();
    let after_reacquire_clear = active_memory_bytes();
    assert!(
        after_reacquire_clear <= baseline,
        "active memory did not return to baseline after re-acquire clear \
         (after={after_reacquire_clear}, baseline={baseline})"
    );

    // Verify `mlx_clear_streams()` also releases a thread's persistent scratch arenas
    // immediately (before thread exit), matching worker-thread teardown. Run on a
    // dedicated thread because `mlx_clear_streams()` destroys the calling thread's
    // default stream.
    let (held_before_clear_streams, after_clear_streams) = std::thread::spawn(|| {
        run_all_inkling2_scratch_paths();
        synchronize();
        let held = active_memory_bytes();
        // SAFETY: stream is synchronized and idle on this dedicated thread.
        unsafe { mlx_sys::mlx_clear_streams() };
        let after = active_memory_bytes();
        (held, after)
    })
    .join()
    .expect("clear_streams worker thread panicked");
    assert!(
        held_before_clear_streams >= baseline + MIN_COMBINED_SCRATCH_BYTES,
        "expected persistent scratch live before clear_streams \
         (held={held_before_clear_streams}, baseline={baseline})"
    );
    assert!(
        after_clear_streams <= baseline,
        "mlx_clear_streams did not release Inkling2 scratch before thread exit \
         (after={after_clear_streams}, baseline={baseline})"
    );
}
