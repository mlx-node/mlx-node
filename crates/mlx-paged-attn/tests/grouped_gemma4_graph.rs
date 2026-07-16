//! Model-free production-graph parity and selector coverage for Gemma 4's
//! staged BF16 D512/BS16/16Q/1KV grouped paged-decode kernel.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

#[test]
fn grouped_gemma4_graph_parity_across_partial_and_stripe_boundaries() {
    // Both values are cached by the C++ dispatcher. This integration-test
    // binary contains no other graph dispatch, so set them before first use.
    unsafe {
        std::env::set_var("MLX_PAGED_GROUPED_GEMMA4", "force");
        std::env::set_var("MLX_PAGED_GROUPED_GEMMA4_TEST_PROBE", "1");
    }

    // Every length ends in a partial physical block. The cases cover the V2
    // floor and each automatic 32/64/128-stripe tier without retaining more
    // than one fixture's buffers at a time.
    for context_len in [513, 3_071, 4_097, 8_193, 16_383] {
        unsafe { mlx_sys::mlx_paged_grouped_gemma4_test_probe_reset() };
        let rc = unsafe { mlx_sys::mlx_paged_grouped_gemma4_graph_parity(context_len) };
        if rc == -3 {
            eprintln!("skipping grouped Gemma 4 graph parity: Metal unavailable");
            return;
        }
        assert_eq!(
            rc, 1,
            "staged Gemma 4 graph parity failed at context {context_len}"
        );
        assert!(
            unsafe { mlx_sys::mlx_paged_grouped_gemma4_test_probe_count() } > 0,
            "context {context_len} silently used generic V2"
        );
    }
}

#[test]
fn grouped_gemma4_selector_boundaries_are_pinned() {
    let selected = |mode, rows, context| unsafe {
        mlx_sys::mlx_paged_grouped_gemma4_shape_guard_for_test(mode, rows, context)
    };

    assert_eq!(selected(0, 1, 3_458), 0, "default/disabled is safe");
    assert_eq!(selected(1, 1, 3_071), 0);
    assert_eq!(selected(1, 1, 3_072), 1);
    assert_eq!(selected(1, 1, 16_384), 1);
    assert_eq!(selected(1, 1, 16_385), 0);

    assert_eq!(
        selected(2, 1, 512),
        0,
        "V1 cannot launch the grouped V2 kernel"
    );
    assert_eq!(selected(2, 1, 513), 1);
    assert_eq!(selected(2, 1, 32_768), 1);
    assert_eq!(selected(2, 2, 3_458), 0, "Gemma q_len=2 remains generic");
}
