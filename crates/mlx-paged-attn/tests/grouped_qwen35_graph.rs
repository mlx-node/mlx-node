//! Model-free graph-native parity for the exact Qwen3.5/3.6 grouped-GQA
//! paged-decode specialization. This is the same C++ Custom primitive path
//! used by `mlx agent`, not only the standalone raw Metal dispatcher.

#![cfg(target_os = "macos")]

#[test]
fn grouped_qwen35_graph_decode_and_mtp_parity() {
    // The probe flag is cached by the C++ dispatcher before its first grouped
    // launch. This integration-test binary has no other graph dispatches, so
    // enabling it here cannot race production work or perturb kernel timing.
    unsafe { std::env::set_var("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE", "1") };
    for query_rows in [1, 2] {
        unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_reset() };
        let rc = unsafe { mlx_sys::mlx_paged_grouped_qwen35_graph_parity(query_rows) };
        if rc == -3 {
            eprintln!("skipping grouped Qwen3.5 graph parity: Metal unavailable");
            return;
        }
        assert_eq!(
            rc, 1,
            "graph-native grouped Qwen3.5 parity failed for query_rows={query_rows}"
        );
        let grouped_launches = unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_count() };
        assert!(
            grouped_launches > 0,
            "graph parity silently used generic V2 for query_rows={query_rows}"
        );
    }
}

#[test]
fn grouped_qwen35_graph_route_thresholds_are_pinned() {
    for (query_rows, below, selected) in [(1, 16_383, 16_384), (2, 8_191, 8_192)] {
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_grouped_qwen35_shape_guard_for_test(query_rows, below) },
            0,
            "grouped route selected below the measured break-even"
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_grouped_qwen35_shape_guard_for_test(query_rows, selected) },
            1,
            "grouped route was not selected at its measured break-even"
        );
    }
}
