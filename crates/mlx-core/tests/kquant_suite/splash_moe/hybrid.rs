struct HybridEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl HybridEnv {
    fn new() -> Self {
        let names = [
            FLAG,
            "MLX_SPLASH_MOE_REGISTER_HYBRID",
            "MLX_SPLASH_MOE_REGISTER_DOWN_BITS",
        ];
        let saved = names.map(|name| (name, std::env::var_os(name))).to_vec();
        unsafe {
            std::env::set_var(FLAG, "1");
            std::env::set_var("MLX_SPLASH_MOE_REGISTER_HYBRID", "1");
            std::env::remove_var("MLX_SPLASH_MOE_REGISTER_DOWN_BITS");
        }
        Self(saved)
    }
}
impl Drop for HybridEnv {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
fn hybrid_flags(register: bool, hybrid: bool) {
    flag(register);
    unsafe {
        std::env::set_var(
            "MLX_SPLASH_MOE_REGISTER_HYBRID",
            if hybrid { "1" } else { "0" },
        )
    };
}
fn hybrid_stage(f: &Fixture, arrays: &[MxArray], stage: i32) -> Vec<u32> {
    assert_eq!(
        arrays.len(),
        21,
        "stage probes do not support device budgets"
    );
    let handles: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
    let result = read_output("hybrid register stage", unsafe {
        mlx_sys::mlx_test_splash_moe_stage(
            handles.as_ptr(),
            handles.len(),
            f.modes.as_ptr(),
            TOP as i32,
            true,
            stage,
        )
    });
    assert_eq!(result.1, DType::BFloat16);
    result.2
}
fn hybrid_routing(arrays: &[MxArray]) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let (coefficients, ids) = topk_from_logits(
        &arrays[1],
        E as i32,
        TOP as i32,
        RoutingMode::Qwen35 {
            renormalize_topk: true,
        },
    )
    .unwrap();
    let coefficients = coefficients
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec();
    let ids = ids.to_uint32().unwrap().to_vec();
    let shared = Activations::sigmoid(&arrays[2])
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec();
    (coefficients, ids, shared)
}

#[test]
#[ignore = "exclusive GPU, hybrid exact per-tile stage selection and independent F64"]
fn register_hybrid_selects_exact_column_and_register_stages() {
    let _guard = SERIAL.lock().unwrap();
    let _env = HybridEnv::new();
    assert!(gpu_gen() >= 17);
    for tile in [false, true] {
        for (down, shared_down) in [
            ("q4k", "q4k"),
            ("q6k", "q6k"),
            ("q4k", "q6k"),
            ("q6k", "q4k"),
        ] {
            let f = Fixture::new(down, shared_down, tile);
            for first_occupancy in [1usize, 2, 3] {
                let mut arrays = f.inputs(8, 2399);
                let mut logits = vec![-8.0f32; 8 * E];
                for row in 0..8 {
                    logits[row * E + usize::from(row >= first_occupancy)] = 4.0;
                    logits[row * E + 2 + row % (E - 2)] = 3.5;
                }
                arrays[1] = MxArray::from_float32(&logits, &[8, E as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let (coefficients, ids, shared) = hybrid_routing(&arrays);
                let mut occupancy = [0usize; E];
                for &expert in &ids {
                    occupancy[expert as usize] += 1;
                }
                assert!(occupancy.contains(&first_occupancy));
                let mut down_bits = Vec::new();
                for stage in 1..=4 {
                    hybrid_flags(false, false);
                    let original = hybrid_stage(&f, &arrays, stage);
                    hybrid_flags(true, false);
                    let register = hybrid_stage(&f, &arrays, stage);
                    hybrid_flags(true, true);
                    let hybrid = hybrid_stage(&f, &arrays, stage);
                    let width = if stage == 4 { H } else { I };
                    for row in 0..8 {
                        for slot in 0..=TOP {
                            let use_column =
                                slot < TOP && occupancy[ids[row * TOP + slot] as usize] <= 2;
                            let reference = if use_column { &original } else { &register };
                            let start = (row * (TOP + 1) + slot) * width;
                            assert_eq!(
                                hybrid[start..start + width],
                                reference[start..start + width],
                                "stage={stage} row={row} slot={slot} column={use_column} first={first_occupancy} tile={tile} down={down}/{shared_down}"
                            );
                        }
                    }
                    if stage == 4 {
                        down_bits = hybrid;
                    }
                }
                start_counting();
                let full = f.call(&arrays);
                assert_eq!(family_count("splash_moe_register_hybrid"), 1);
                stop_counting();
                // Combine the independently selected stage outputs using the
                // unchanged BF16 coefficient/product/reduction boundaries.
                let mut combined = vec![0.0f32; 8 * H];
                for row in 0..8 {
                    for column in 0..H {
                        let mut routed = 0.0;
                        for slot in 0..TOP {
                            let v = half::bf16::from_bits(
                                down_bits[(row * (TOP + 1) + slot) * H + column] as u16,
                            )
                            .to_f32();
                            routed = bf(routed + bf(coefficients[row * TOP + slot] * v));
                        }
                        let v = half::bf16::from_bits(
                            down_bits[(row * (TOP + 1) + TOP) * H + column] as u16,
                        )
                        .to_f32();
                        combined[row * H + column] = bf(routed + bf(shared[row] * v));
                    }
                }
                assert_eq!(full, combined, "hybrid combine changed its dtype/order");
                let oracle = f.oracle(&arrays);
                for budget in [1i32, 3, 8] {
                    arrays.truncate(21);
                    arrays.push(MxArray::from_int32(&[budget], &[]).unwrap());
                    let actual = f.call(&arrays);
                    let live = budget as usize * H;
                    assert!(actual.iter().all(|v| v.is_finite()));
                    assert!(actual[live..].iter().all(|v| v.to_bits() == 0));
                    let peak = oracle[..live]
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0f32, f32::max);
                    let error = actual[..live]
                        .iter()
                        .zip(&oracle[..live])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let prefix_error = actual[..live]
                        .iter()
                        .zip(&full[..live])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "hybrid tile={tile} down={down}/{shared_down} first={first_occupancy} budget={budget} oracle_max_peak={} prefix_max_peak={}",
                        error / peak.max(1e-30),
                        prefix_error / peak.max(1e-30)
                    );
                    assert!(
                        error <= peak * 0.01 + 1e-6,
                        "independent F64 threshold unchanged"
                    );
                    assert!(
                        prefix_error <= peak * 0.01 + 1e-6,
                        "occupancy transition exceeds the established bound"
                    );
                    if budget == 8 {
                        assert_eq!(actual, full);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "exclusive GPU, hybrid Q4/Q6 layer selection and unsupported-width fallback"]
fn register_hybrid_format_filter_and_fallbacks_are_exact() {
    let _guard = SERIAL.lock().unwrap();
    let _env = HybridEnv::new();
    assert!(gpu_gen() >= 17);
    for down in ["q4k", "q6k"] {
        let f = Fixture::new(down, if down == "q4k" { "q6k" } else { "q4k" }, true);
        for rows in [1, 8, 33] {
            let arrays = f.inputs(rows, 2503);
            hybrid_flags(false, false);
            let original = f.call(&arrays);
            hybrid_flags(true, true);
            unsafe { std::env::remove_var("MLX_SPLASH_MOE_REGISTER_DOWN_BITS") };
            let hybrid = f.call(&arrays);
            for filter in ["4", "6", "invalid"] {
                unsafe { std::env::set_var("MLX_SPLASH_MOE_REGISTER_DOWN_BITS", filter) };
                let selected = rows == 8
                    && ((filter == "4" && down == "q4k") || (filter == "6" && down == "q6k"));
                start_counting();
                let result = f.call(&arrays);
                assert_eq!(
                    family_count("splash_moe_register_hybrid"),
                    u64::from(selected)
                );
                assert_eq!(
                    family_count("splash_moe_register_gate_up"),
                    u64::from(selected)
                );
                stop_counting();
                assert_eq!(&result, if selected { &hybrid } else { &original });
            }
        }
    }
}
