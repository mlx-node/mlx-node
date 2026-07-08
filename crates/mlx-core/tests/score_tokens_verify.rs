//! Unit tests for the pure `scoreTokens` verify helpers in
//! `crate::inspector` — position mapping, top-K selection, and the accept
//! rule. No model weights required: the helpers are deliberately factored
//! free of MxArray / model state so the off-by-one at the prefix/draft
//! boundary is testable in isolation.
//!
//! Lives as an integration test (not an in-crate `#[cfg(test)]` module)
//! because this branch's lib-test target has pre-existing unrelated
//! compile breakage (`crate::test_support` is not declared in lib.rs).
//!
//! Run with:
//!
//! ```shell
//! cargo test -p mlx-core --test score_tokens_verify
//! ```

use mlx_core::inspector::{
    top_k_from_row, top_k_rows, verify_accept_flags, verify_logit_row_bounds,
};

/// Link-time stubs for C++ FFI shims that are declared in
/// `mlx-sys/src/lib.rs` but NOT implemented by this branch's C++ bridge
/// (`mlx_qwen35.cpp` here lacks the VLM / MoE / paged shims that
/// `origin/main` ships) — pre-existing breakage tracked as the
/// "broken native addon" state (#68). Without these, NO mlx-core test
/// binary links natively, because napi's registration ctors keep every
/// model path reachable.
///
/// The stubs exist purely to satisfy the linker; none of the tests below
/// go anywhere near the VLM / MoE / paged code, so they can never be
/// called. If the real shims land later, these will produce duplicate-
/// symbol errors — delete this module then.
mod missing_native_shim_stubs {
    macro_rules! stub {
        ($($name:ident),+ $(,)?) => {
            $(
                #[unsafe(no_mangle)]
                pub extern "C" fn $name() {
                    unreachable!(concat!(
                        stringify!($name),
                        " is a link-time stub (see score_tokens_verify.rs); it must never be called"
                    ));
                }
            )+
        };
    }

    stub!(
        mlx_qwen35_export_paged_linear_caches,
        mlx_qwen35_forward_paged,
        mlx_qwen35_get_paged_cache_offset,
        mlx_qwen35_init_paged,
        mlx_qwen35_moe_adjust_offset,
        mlx_qwen35_moe_eval_token_and_caches,
        mlx_qwen35_moe_export_caches,
        mlx_qwen35_moe_export_paged_linear_caches,
        mlx_qwen35_moe_forward,
        mlx_qwen35_moe_forward_paged,
        mlx_qwen35_moe_get_cache_offset,
        mlx_qwen35_moe_get_paged_cache_offset,
        mlx_qwen35_moe_init_from_prefill,
        mlx_qwen35_moe_init_paged,
        mlx_qwen35_moe_reset,
        mlx_qwen35_vlm_cache_count,
        mlx_qwen35_vlm_get_cache,
        mlx_qwen35_vlm_prefill,
        mlx_qwen35_vlm_reset,
    );
}

/// Build a fake row-major `[T, vocab]` logits buffer where row `r` has its
/// maximum at token id `argmax_of_row(r)` and strictly smaller, distinct
/// values for the remaining ids, so top-K order is fully deterministic.
fn fake_logits(t: usize, vocab: usize, argmax_of_row: impl Fn(usize) -> u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; t * vocab];
    for r in 0..t {
        let peak = argmax_of_row(r) as usize;
        assert!(peak < vocab);
        for v in 0..vocab {
            buf[r * vocab + v] = if v == peak { 100.0 } else { -(v as f32) - 1.0 };
        }
    }
    buf
}

#[test]
fn top_k_from_row_orders_descending() {
    let row = [1.0f32, 5.0, 3.0, 2.0];
    let (ids, logits) = top_k_from_row(&row, 2);
    assert_eq!(ids, vec![1, 2]);
    assert_eq!(logits, vec![5.0, 3.0]);
    // K larger than the row clamps instead of panicking.
    let (ids_all, logits_all) = top_k_from_row(&row, 99);
    assert_eq!(ids_all, vec![1, 2, 3, 0]);
    assert_eq!(logits_all, vec![5.0, 3.0, 2.0, 1.0]);
    // K == 0 and empty rows yield empty outputs.
    assert_eq!(top_k_from_row(&row, 0), (Vec::new(), Vec::new()));
    assert_eq!(top_k_from_row(&[], 3), (Vec::new(), Vec::new()));
}

#[test]
fn top_k_rows_validates_shape() {
    // 2 rows x 3 vocab buffer with only 5 floats → loud error.
    let bad = [0.0f32; 5];
    assert!(top_k_rows(&bad, 2, 3, 1).is_err());
    assert!(top_k_rows(&bad, 5, 0, 1).is_err());
    assert!(top_k_rows(&bad, 5, 1, 0).is_err());
    // Well-formed buffer: per-row results in row order.
    let buf = [0.0f32, 9.0, 1.0, /* row 1 */ 7.0, 0.0, 3.0];
    let rows = top_k_rows(&buf, 2, 3, 2).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, vec![1, 2]);
    assert_eq!(rows[1].0, vec![0, 2]);
}

#[test]
fn verify_bounds_rejects_empty_inputs() {
    assert!(verify_logit_row_bounds(0, 3).is_err());
    assert!(verify_logit_row_bounds(4, 0).is_err());
    assert_eq!(verify_logit_row_bounds(1, 1).unwrap(), (0, 1));
    assert_eq!(verify_logit_row_bounds(4, 3).unwrap(), (3, 6));
}

/// End-to-end over a fake `[1, T, vocab]` buffer: the D selected rows must
/// be `prefix_len - 1 + i` (NOT `prefix_len + i`), and the argmax / accept
/// flags must reflect exactly those rows — this is the off-by-one at the
/// prefix/draft boundary the widget depends on.
#[test]
fn verify_positions_and_accept_rule_at_boundary() {
    let vocab = 7usize;
    let prefix_len = 4usize;
    let draft_len = 3usize;
    let t = prefix_len + draft_len; // T = 7
    // Row r predicts token id (2 * r) % vocab.
    let argmax_of_row = |r: usize| ((2 * r) % vocab) as u32;
    let buf = fake_logits(t, vocab, argmax_of_row);

    let (start, end) = verify_logit_row_bounds(prefix_len, draft_len).unwrap();
    assert_eq!((start, end), (3, 6));
    // Last used row is T - 2: row T - 1 (predicting the post-draft token)
    // must be excluded.
    assert_eq!(end, (t as i64) - 1);

    // Host-side equivalent of the on-device slice_axis(1, start, end).
    let rows = &buf[(start as usize) * vocab..(end as usize) * vocab];
    let per_row = top_k_rows(rows, draft_len, vocab, 3).unwrap();

    // Expected argmax per verify position i comes from row
    // prefix_len - 1 + i → rows 3, 4, 5 → ids 6, 1, 3.
    let argmax_ids: Vec<u32> = per_row.iter().map(|(ids, _)| ids[0]).collect();
    assert_eq!(argmax_ids, vec![6, 1, 3]);

    // Draft agrees at positions 0 and 2, disagrees at 1.
    let draft_ids = [6u32, 2, 3];
    assert_eq!(
        verify_accept_flags(&argmax_ids, &draft_ids),
        vec![true, false, true]
    );

    // Sanity: had the mapping been off by one (rows 4..7), position 0
    // would have argmax 1, not 6.
    assert_eq!(argmax_of_row(prefix_len), 1);
    assert_ne!(argmax_ids[0], argmax_of_row(prefix_len));
}
