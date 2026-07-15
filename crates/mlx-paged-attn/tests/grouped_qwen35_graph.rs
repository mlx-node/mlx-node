//! Model-free graph-native parity for the exact Qwen3.5/3.6 grouped-GQA
//! paged-decode specialization. This is the same C++ Custom primitive path
//! used by `mlx agent`, not only the standalone raw Metal dispatcher.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

#[test]
fn grouped_qwen35_graph_decode_and_mtp_parity() {
    // The probe flag is cached by the C++ dispatcher before its first grouped
    // launch. This integration-test binary has no other graph dispatches, so
    // enabling it here cannot race production work or perturb kernel timing.
    unsafe { std::env::set_var("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE", "1") };
    for (num_q_heads, num_kv_heads) in [(24, 4), (16, 2)] {
        for query_rows in [1, 2] {
            unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_reset() };
            let rc = unsafe {
                mlx_sys::mlx_paged_grouped_qwen35_graph_parity(
                    query_rows,
                    num_q_heads,
                    num_kv_heads,
                )
            };
            if rc == -3 {
                eprintln!("skipping grouped Qwen3.5 graph parity: Metal unavailable");
                return;
            }
            assert_eq!(
                rc, 1,
                "graph-native grouped Qwen3.5 parity failed for \
                 q={num_q_heads} kv={num_kv_heads} query_rows={query_rows}"
            );
            let grouped_launches = unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_count() };
            assert!(
                grouped_launches > 0,
                "graph parity silently used generic V2 for \
                 q={num_q_heads} kv={num_kv_heads} query_rows={query_rows}"
            );
        }
    }
}

#[test]
fn grouped_qwen35_graph_route_thresholds_are_pinned() {
    for (num_q_heads, num_kv_heads, decode_min, verify_min) in
        [(24, 4, 16_384, 8_192), (16, 2, 32_768, 16_384)]
    {
        for (query_rows, selected) in [(1, decode_min), (2, verify_min)] {
            let below = selected - 1;
            assert_eq!(
                unsafe {
                    mlx_sys::mlx_paged_grouped_qwen35_shape_guard_for_test(
                        num_q_heads,
                        num_kv_heads,
                        query_rows,
                        below,
                    )
                },
                0,
                "grouped route selected below its configured threshold"
            );
            assert_eq!(
                unsafe {
                    mlx_sys::mlx_paged_grouped_qwen35_shape_guard_for_test(
                        num_q_heads,
                        num_kv_heads,
                        query_rows,
                        selected,
                    )
                },
                1,
                "grouped route was not selected at its configured threshold"
            );
        }
    }
    for (num_q_heads, num_kv_heads) in [(16, 4), (24, 2)] {
        assert_eq!(
            unsafe {
                mlx_sys::mlx_paged_grouped_qwen35_shape_guard_for_test(
                    num_q_heads,
                    num_kv_heads,
                    1,
                    16_384,
                )
            },
            0,
            "unsupported nearby head shape selected the grouped route"
        );
    }
}
