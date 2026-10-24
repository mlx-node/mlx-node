use super::*;
const FLAG: &str = "MLX_SPLASH_MOE_PAIRED_STAGE";
struct Env(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl Env {
    fn set(values: &[(&'static str, &str)]) -> Self {
        Self(
            values
                .iter()
                .map(|&(name, value)| {
                    let old = std::env::var_os(name);
                    unsafe { std::env::set_var(name, value) };
                    (name, old)
                })
                .collect(),
        )
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        for (name, old) in &self.0 {
            unsafe {
                match old {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
/// Used by the saved-checkpoint stage probe too. Every off/on primitive is
/// freshly constructed; no environment toggle crosses a cached compiled graph.
pub(super) fn maybe_compare(
    name: &str,
    mut build: impl FnMut() -> *mut mlx_sys::mlx_array,
) -> (Vec<i64>, DType, Vec<u32>) {
    let result = read_output(name, build());
    if std::env::var_os("SPLASH_MOE_PAIRED_COMPARE").is_some() {
        let _restore = Env::set(&[(FLAG, "0")]);
        let reference = read_output(name, build());
        unsafe { std::env::set_var(FLAG, "1") };
        start_counting();
        let paired = read_output(name, build());
        assert!(family_count("splash_moe_paired_stage_enabled") > 0);
        stop_counting();
        assert_eq!(
            paired, reference,
            "{name}: paired high/low must preserve every output bit"
        );
        assert_eq!(
            result, reference,
            "{name}: original setting must match either schedule"
        );
    }
    result
}

#[test]
#[ignore = "exclusive GPU, exact high/low staging schedule ablation"]
fn paired_stage_synthetic_boundaries_and_budget_are_bit_exact() {
    let _guard = SERIAL.lock().unwrap();
    assert!(unsafe { mlx_sys::mlx_splash_moe_available() });
    let _env = Env::set(&[
        (FLAG, "0"),
        ("MLX_SPLASH_MOE_REGISTER", "0"),
        ("SPLASH_MOE_PAIRED_COMPARE", "1"),
    ]);
    for names in [
        ["q4k", "q4k", "q6k", "q4k", "q4k", "q6k"],
        ["q5k", "q5k", "q5k", "q5k", "q5k", "q5k"],
        ["q6k", "q4k", "q4k", "q4k", "q6k", "q6k"],
    ] {
        let formats: Vec<_> = names.into_iter().map(kquant).collect();
        let raw: Vec<_> = formats
            .iter()
            .enumerate()
            .map(|(j, q)| {
                let (n, k) = if j % 3 == 2 { (H, I) } else { (I, H) };
                let shape = if j < 3 {
                    vec![E as i64, n as i64]
                } else {
                    vec![n as i64]
                };
                let mut w = weights(q, &shape, k as i64, 0xDD00 + j as u32);
                w.biases = w
                    .biases
                    .mul_scalar(0.001)
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap();
                w
            })
            .collect();
        for tile in [false, true] {
            let packed: Vec<_> = raw
                .iter()
                .zip(&formats)
                .map(|(w, q)| Weights {
                    w: if tile {
                        tiled(&w.w, q.bits as i64)
                    } else {
                        w.w.clone()
                    },
                    scales: if tile {
                        tiled(&w.scales, 16)
                    } else {
                        w.scales.clone()
                    },
                    biases: if tile {
                        tiled(&w.biases, q.biases_cols)
                    } else {
                        w.biases.clone()
                    },
                })
                .collect();
            let modes: Vec<_> = formats
                .iter()
                .map(|q| q.bits as u32 | if tile { 256 } else { 0 })
                .collect();
            let cases: Vec<_> = [1i64, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33]
                .into_iter()
                .map(|m| (m, m, m))
                .chain([1, 2, 3, 7, 8].into_iter().map(|n| (8, n, 8)))
                .chain([1, 3].into_iter().map(|n| (8, 8, n)))
                .collect();
            for (m, occupancy, budget) in cases {
                let x = activation(&[m, H as i64], 910 + m as u32, DType::BFloat16);
                let mut logits = vec![-9.0; m as usize * E];
                for r in 0..m as usize {
                    let first = if r < occupancy as usize {
                        0
                    } else {
                        1 + r % (E - 1)
                    };
                    let second = 1 + (r + 1) % (E - 1);
                    let second = if second == first {
                        1 + (r + 2) % (E - 1)
                    } else {
                        second
                    };
                    logits[r * E + first] = 4.0;
                    logits[r * E + second] = 3.0;
                }
                let router = MxArray::from_float32(&logits, &[m, E as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let shared = activation(&[m, 1], 912, DType::BFloat16);
                let budget_array = MxArray::from_int32(&[budget as i32], &[]).unwrap();
                let mut arrays = vec![x.as_raw_ptr(), router.as_raw_ptr(), shared.as_raw_ptr()];
                for w in &packed {
                    arrays.extend([
                        w.w.as_raw_ptr(),
                        w.scales.as_raw_ptr(),
                        w.biases.as_raw_ptr(),
                    ]);
                }
                if budget < m {
                    arrays.push(budget_array.as_raw_ptr());
                }
                // Stage scatters cover all route slots and are intentionally unbudgeted.
                for stage in 1..=if budget == m { 4 } else { 0 } {
                    let _ = maybe_compare("synthetic paired stage", || unsafe {
                        mlx_sys::mlx_test_splash_moe_stage(
                            arrays.as_ptr(),
                            arrays.len(),
                            modes.as_ptr(),
                            TOP as i32,
                            true,
                            stage,
                        )
                    });
                }
                let _ = maybe_compare("synthetic paired combined", || unsafe {
                    mlx_sys::mlx_splash_moe(
                        arrays.as_ptr(),
                        arrays.len(),
                        modes.as_ptr(),
                        TOP as i32,
                        true,
                    )
                });
            }
        }
    }
}

#[test]
#[ignore = "exclusive GPU, diagnostic stage budget rejection"]
fn budgeted_stage_probe_is_rejected_and_combined_output_stays_supported() {
    let _guard = SERIAL.lock().unwrap();
    assert!(unsafe { mlx_sys::mlx_splash_moe_available() });
    let _env = Env::set(&[("MLX_SPLASH_MOE_REGISTER", "0")]);
    let q = kquant("q4k");
    let packed: Vec<_> = (0..6)
        .map(|j| {
            weights(
                q,
                &if j < 3 { vec![2, 256] } else { vec![256] },
                256,
                200 + j,
            )
        })
        .collect();
    let x = activation(&[8, 256], 211, DType::BFloat16);
    let router = activation(&[8, 2], 212, DType::BFloat16);
    let shared = activation(&[8, 1], 213, DType::BFloat16);
    let budget = MxArray::from_int32(&[1], &[]).unwrap();
    let mut inputs = vec![x.as_raw_ptr(), router.as_raw_ptr(), shared.as_raw_ptr()];
    for w in &packed {
        inputs.extend([
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
        ]);
    }
    inputs.push(budget.as_raw_ptr());
    for setting in ["0", "1"] {
        let _env = Env::set(&[(FLAG, setting)]);
        for stage in 1..=4 {
            let result = unsafe {
                mlx_sys::mlx_test_splash_moe_stage(
                    inputs.as_ptr(),
                    inputs.len(),
                    [4u32; 6].as_ptr(),
                    1,
                    true,
                    stage,
                )
            };
            assert!(
                result.is_null(),
                "budgeted diagnostic stage{stage} must reject before graph execution"
            );
        }
        let result = read_output("budgeted production combined", unsafe {
            mlx_sys::mlx_splash_moe(inputs.as_ptr(), inputs.len(), [4u32; 6].as_ptr(), 1, true)
        });
        assert_eq!(result.0, vec![8, 256]);
        assert!(result.2[256..].iter().all(|&v| v == 0));
    }
    // Register experts take precedence and ignore the staged-schedule flag.
    let _register = Env::set(&[("MLX_SPLASH_MOE_REGISTER", "1"), (FLAG, "0")]);
    let call = || unsafe {
        mlx_sys::mlx_splash_moe(inputs.as_ptr(), inputs.len(), [4u32; 6].as_ptr(), 1, true)
    };
    let original = read_output("register precedence off", call());
    unsafe { std::env::set_var(FLAG, "1") };
    start_counting();
    let paired = read_output("register precedence on", call());
    assert_eq!(family_count("splash_moe_register_gate_up"), 1);
    assert_eq!(family_count("splash_moe_paired_stage_enabled"), 0);
    stop_counting();
    assert_eq!(paired, original);
}
