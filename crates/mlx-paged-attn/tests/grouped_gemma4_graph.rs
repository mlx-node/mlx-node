//! Model-free production-graph parity and selector coverage for the direct-read
//! BF16 D512/BS16 grouped paged-decode kernel.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

#[test]
fn grouped_d512_graph_parity_across_geometries_and_stripe_boundaries() {
    // Both values are cached by the C++ dispatcher. This integration-test
    // binary contains no other graph dispatch, so set them before first use.
    unsafe {
        std::env::set_var("MLX_PAGED_GROUPED_D512", "force");
        std::env::set_var("MLX_PAGED_GROUPED_D512_TEST_PROBE", "1");
    }

    for (num_q_heads, num_kv_heads) in [(8, 1), (16, 1), (16, 2), (32, 4)] {
        // Exercise every stripe boundary for the Hkv2 production target and
        // the V2 floor plus a long-context tier for the other shipped
        // geometries. Every case ends in a partial physical page.
        let contexts: &[i32] = if (num_q_heads, num_kv_heads) == (16, 2) {
            &[513, 3_071, 4_097, 8_193, 16_383]
        } else {
            &[513, 8_193]
        };
        for &context_len in contexts {
            unsafe { mlx_sys::mlx_paged_grouped_d512_test_probe_reset() };
            let rc = unsafe {
                mlx_sys::mlx_paged_grouped_d512_graph_parity(num_q_heads, num_kv_heads, context_len)
            };
            if rc == -3 {
                eprintln!("skipping grouped D512 graph parity: Metal unavailable");
                return;
            }
            assert_eq!(
                rc, 1,
                "direct-read D512 graph parity failed for q={num_q_heads} \
                 kv={num_kv_heads} at context {context_len}"
            );
            assert!(
                unsafe { mlx_sys::mlx_paged_grouped_d512_test_probe_count() } > 0,
                "q={num_q_heads} kv={num_kv_heads} context={context_len} \
                 silently used generic V2"
            );
        }
    }
}

#[test]
fn grouped_d512_selector_boundaries_and_geometries_are_pinned() {
    let selected = |mode, q_heads, kv_heads, rows, context| unsafe {
        mlx_sys::mlx_paged_grouped_d512_shape_guard_for_test(mode, q_heads, kv_heads, rows, context)
    };

    for (q_heads, kv_heads) in [(8, 1), (16, 1), (16, 2), (32, 4)] {
        assert_eq!(
            selected(0, q_heads, kv_heads, 1, 3_458),
            0,
            "default/disabled is safe"
        );
        assert_eq!(selected(1, q_heads, kv_heads, 1, 3_071), 0);
        let legacy_auto = i32::from((q_heads, kv_heads) == (16, 1));
        assert_eq!(selected(1, q_heads, kv_heads, 1, 3_072), legacy_auto);
        assert_eq!(selected(1, q_heads, kv_heads, 1, 16_384), legacy_auto);
        assert_eq!(selected(1, q_heads, kv_heads, 1, 16_385), 0);

        assert_eq!(
            selected(2, q_heads, kv_heads, 1, 512),
            0,
            "V1 cannot launch the grouped V2 kernel"
        );
        assert_eq!(selected(2, q_heads, kv_heads, 1, 513), 1);
        assert_eq!(selected(2, q_heads, kv_heads, 1, 32_768), 1);
        assert_eq!(
            selected(2, q_heads, kv_heads, 2, 3_458),
            0,
            "q_len=2 remains generic"
        );
    }

    for (q_heads, kv_heads) in [(8, 2), (16, 4), (24, 4), (32, 2)] {
        assert_eq!(
            selected(2, q_heads, kv_heads, 1, 8_193),
            0,
            "unsupported nearby geometry selected grouped D512"
        );
    }

    // The original interface remains an exact compatibility wrapper.
    assert_eq!(
        unsafe { mlx_sys::mlx_paged_grouped_gemma4_shape_guard_for_test(2, 1, 513) },
        selected(2, 16, 1, 1, 513)
    );
}

#[test]
fn grouped_d512_graph_stripe_policy_matches_measured_boundaries() {
    let stripes = |q_heads, kv_heads, context, override_stripes| unsafe {
        mlx_sys::mlx_paged_grouped_d512_stripe_count_for_test(
            q_heads,
            kv_heads,
            context,
            override_stripes,
        )
    };

    for (context, expected) in [(90_112, 64), (90_113, 128), (91_795, 128), (112_000, 128)] {
        assert_eq!(stripes(16, 2, context, 0), expected);
    }
    assert_eq!(stripes(8, 1, 90_113, 0), 128, "Hq8/Hkv1 is unchanged");
    assert_eq!(stripes(16, 1, 90_113, 0), 128, "Hq16/Hkv1 is unchanged");
    assert_eq!(stripes(32, 4, 90_113, 0), 32, "Hkv4 is unchanged");
    assert_eq!(
        stripes(16, 2, 91_795, 32),
        32,
        "an explicit validated override remains authoritative"
    );
}
