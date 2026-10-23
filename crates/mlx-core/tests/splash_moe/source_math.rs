use super::*;

pub(super) fn require_source_bits(directory: &std::path::Path, name: &str, actual: &[u32]) {
    let bytes = std::fs::read(directory.join(name)).unwrap();
    assert_eq!(bytes.len(), actual.len() * 2, "source stage length: {name}");
    let expected: Vec<u32> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
        .collect();
    let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    eprintln!(
        "SOURCE_BITS stage={name} elements={} mismatches={mismatches}",
        actual.len()
    );
    assert_eq!(
        mismatches, 0,
        "source stage mismatch must be explained before changing the gate: {name}"
    );
}

#[test]
#[ignore = "exclusive GPU, exact original-GGUF source replay stages on captured M8 operands"]
fn source_math_actual_layer_matches_source_replay() {
    assert_eq!(
        std::env::var("MLX_SPLASH_MOE_SOURCE_MATH").as_deref(),
        Ok("1")
    );
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_ORACLE_DIR").is_some());
    grouped_actual_layer_stage_comparison();
}

use std::sync::atomic::{AtomicUsize, Ordering};
struct Builder {
    policy: Option<bool>,
    modes: Vec<u32>,
    traces: AtomicUsize,
}
unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if context.is_null() || n != 22 || m != 1 {
        return false;
    }
    let b = unsafe { &*context.cast::<Builder>() };
    b.traces.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let out = unsafe {
        match b.policy {
            Some(policy) => mlx_sys::mlx_splash_moe_with_policy(
                a.as_ptr(),
                n,
                b.modes.as_ptr(),
                8,
                true,
                policy,
            ),
            None => mlx_sys::mlx_splash_moe(a.as_ptr(), n, b.modes.as_ptr(), 8, true),
        }
    };
    for h in a {
        unsafe { mlx_sys::mlx_array_delete(h) }
    }
    unsafe { *outputs = out };
    !out.is_null()
}
fn set_flag(enabled: bool) {
    unsafe {
        std::env::set_var(
            "MLX_SPLASH_MOE_SOURCE_MATH",
            if enabled { "1" } else { "0" },
        )
    }
}
fn call(a: &[MxArray], modes: &[u32]) -> (Vec<i64>, DType, Vec<u32>) {
    let handles: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
    read_output("source math replay", unsafe {
        mlx_sys::mlx_splash_moe(handles.as_ptr(), handles.len(), modes.as_ptr(), 8, true)
    })
}
struct ResetFlag(Option<std::ffi::OsString>);
impl Drop for ResetFlag {
    fn drop(&mut self) {
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var("MLX_SPLASH_MOE_SOURCE_MATH", v),
                None => std::env::remove_var("MLX_SPLASH_MOE_SOURCE_MATH"),
            }
        }
    }
}
/// Reuse the real eligible weights; no second model or broad dequantization.
pub(super) fn replay_and_fallbacks(saved: &[MxArray], modes: &[u32]) {
    assert_eq!(saved.len(), 21);
    let _reset = ResetFlag(std::env::var_os("MLX_SPLASH_MOE_SOURCE_MATH"));
    let mut b = Builder {
        policy: None,
        modes: modes.to_vec(),
        traces: AtomicUsize::new(0),
    };
    let key = 0x7372634d6f453801;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    for (index, m) in [8i64, 1, 7, 33, 8].into_iter().enumerate() {
        let mut a = saved.to_vec();
        for j in 0..3 {
            let copies = vec![&saved[j]; ((m + 7) / 8) as usize];
            a[j] = MxArray::concatenate_many(copies, Some(0))
                .unwrap()
                .slice(&[0, 0], &[m, saved[j].shape().unwrap()[1]])
                .unwrap();
        }
        a[0] = a[0]
            .mul_scalar(1.0 + index as f64 * 0.01)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        a.push(MxArray::from_int32(&[m as i32], &[]).unwrap());
        set_flag(false);
        let off = call(&a, modes);
        set_flag(true);
        let expected = call(&a, modes);
        if m != 8 {
            assert_eq!(expected, off, "M{m} source fallback must be bit exact");
        }
        let handles: Vec<_> = a.iter().map(|v| v.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut b as *mut Builder).cast(),
                handles.as_ptr(),
                handles.len(),
                &mut out,
                1,
                true,
            )
        });
        start_counting();
        let got = read_output("compiled source-math width", out);
        assert_eq!(family_count("splash_moe_source_math"), u64::from(m == 8));
        assert_eq!(family_count("splash_moe_source_column"), u64::from(m == 8));
        stop_counting();
        assert_eq!(got, expected);
        assert_eq!(b.traces.load(Ordering::Relaxed), 1);
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    for dtypes in [
        [DType::Float32, DType::BFloat16],
        [DType::BFloat16, DType::Float32],
        [DType::Float32, DType::Float32],
    ] {
        let mut a = saved.to_vec();
        a[1] = a[1].astype(dtypes[0]).unwrap();
        a[2] = a[2].astype(dtypes[1]).unwrap();
        set_flag(false);
        let off = call(&a, modes);
        set_flag(true);
        start_counting();
        let on = call(&a, modes);
        assert_eq!(family_count("splash_moe_source_math"), 0);
        assert_eq!(family_count("splash_moe_source_column"), 0);
        stop_counting();
        assert_eq!(on, off, "unsupported router/shared dtype fallback");
    }
    for j in 0..6 {
        let mut a = saved.to_vec();
        let mut f = modes.to_vec();
        f[j] = (f[j] & 256) | 5;
        let mut shape = a[3 + j * 3].shape().unwrap().to_vec();
        let n = shape.len();
        let k = if j % 3 == 2 { 512 } else { 2048 };
        shape[n - 1] = k * 5 / 32;
        // All-zero Q5 planes are valid, finite weights and allocate one changed
        // projection at a time. This isolates format eligibility, not accuracy.
        a[3 + j * 3] = MxArray::zeros(&shape, Some(DType::Uint32)).unwrap();
        a[4 + j * 3] = MxArray::zeros(&a[4 + j * 3].shape().unwrap(), Some(DType::Uint8)).unwrap();
        shape[n - 1] = k / 256 * 2;
        a[5 + j * 3] = MxArray::zeros(&shape, Some(DType::Float16)).unwrap();
        set_flag(false);
        let off = call(&a, &f);
        set_flag(true);
        start_counting();
        let on = call(&a, &f);
        assert_eq!(family_count("splash_moe_source_math"), 0);
        assert_eq!(family_count("splash_moe_source_column"), 0);
        stop_counting();
        assert_eq!(on, off, "unsupported Q5 projection{j} fallback");
    }
    explicit_policy_replay(saved, modes);
    actual_residual_replay(saved, modes);
    if std::env::var_os("SPLASH_MOE_SOURCE_BUDGET_ORACLE_DIR").is_some() {
        source_budget_replay(saved, modes);
    }
    if std::env::var_os("SPLASH_MOE_SOURCE_RESIDUAL_ORACLE_DIR").is_some() {
        source_residual_oracle_replay(saved, modes);
    }
}

/// The loaded model's explicit policy must win over later environment changes.
/// Reuse one resident checkpoint and one trace for each immutable policy.
fn explicit_policy_replay(saved: &[MxArray], modes: &[u32]) {
    for policy in [false, true] {
        let mut b = Builder {
            policy: Some(policy),
            modes: modes.to_vec(),
            traces: AtomicUsize::new(0),
        };
        let key = 0x737263506f6c3800 + u64::from(policy);
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        for (i, environment) in [!policy, policy, !policy, policy].into_iter().enumerate() {
            let mut a = saved.to_vec();
            a[0] = a[0]
                .mul_scalar(1.0 + i as f64 * 0.02)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            a.push(MxArray::from_int32(&[8], &[]).unwrap());
            set_flag(policy);
            let expected = call(&a, modes);
            set_flag(environment);
            let handles: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
            let explicit = read_output("explicit source policy", unsafe {
                mlx_sys::mlx_splash_moe_with_policy(
                    handles.as_ptr(),
                    handles.len(),
                    modes.as_ptr(),
                    8,
                    true,
                    policy,
                )
            });
            assert_eq!(
                explicit, expected,
                "explicit policy={policy} env={environment}"
            );
            let const_handles: Vec<_> = handles.iter().map(|h| h.cast_const()).collect();
            let mut out = std::ptr::null_mut();
            assert!(unsafe {
                mlx_sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build),
                    (&mut b as *mut Builder).cast(),
                    const_handles.as_ptr(),
                    const_handles.len(),
                    &mut out,
                    1,
                    true,
                )
            });
            start_counting();
            let compiled = read_output("compiled immutable source policy", out);
            assert_eq!(family_count("splash_moe_source_math"), u64::from(policy));
            assert_eq!(family_count("splash_moe_source_column"), u64::from(policy));
            stop_counting();
            assert_eq!(compiled, expected);
            assert_eq!(b.traces.load(Ordering::Relaxed), 1);
        }
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}

struct ResidualBuilder {
    budget: bool,
    modes: Vec<u32>,
    traces: AtomicUsize,
}
unsafe extern "C" fn build_residual(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if context.is_null() || m != 1 {
        return false;
    }
    let b = unsafe { &*context.cast::<ResidualBuilder>() };
    if n != 22 + usize::from(b.budget) {
        return false;
    }
    b.traces.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let output = unsafe {
        mlx_sys::mlx_splash_moe_with_residual(
            a.as_ptr(),
            n - 1,
            b.modes.as_ptr(),
            8,
            true,
            a[n - 1],
        )
    };
    for input in a {
        unsafe { mlx_sys::mlx_array_delete(input) };
    }
    unsafe { *outputs = output };
    !output.is_null()
}

/// The entire production primitive, including optional-budget/residual binding,
/// must keep Hidden output semantics when one shapeless graph leaves M8.
fn actual_residual_replay(saved: &[MxArray], modes: &[u32]) {
    let _reset = ResetFlag(std::env::var_os("MLX_SPLASH_MOE_SOURCE_MATH"));
    for budget in [false, true] {
        let mut builder = ResidualBuilder {
            budget,
            modes: modes.to_vec(),
            traces: AtomicUsize::new(0),
        };
        let key = 0x7372635265733800 + u64::from(budget);
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
        for (iteration, m) in [8i64, 1, 7, 33, 8].into_iter().enumerate() {
            let mut arrays = saved.to_vec();
            for j in 0..3 {
                arrays[j] =
                    MxArray::concatenate_many(vec![&saved[j]; ((m + 7) / 8) as usize], Some(0))
                        .unwrap()
                        .slice(&[0, 0], &[m, saved[j].shape().unwrap()[1]])
                        .unwrap();
            }
            arrays[0] = arrays[0]
                .mul_scalar(1.0 + iteration as f64 * 0.02)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            // Reuse activation storage as read-only residual to cover operand aliases.
            let residual = arrays[0].clone();
            let before = residual.to_float32().unwrap().to_vec();
            let live = if budget && iteration > 1 { m.min(3) } else { m };
            if budget {
                arrays.push(MxArray::from_int32(&[live as i32], &[]).unwrap());
            }
            let handles: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
            set_flag(false); // explicit residual factory must not reread this flag.
            let eager = read_output("actual source residual eager", unsafe {
                mlx_sys::mlx_splash_moe_with_residual(
                    handles.as_ptr(),
                    handles.len(),
                    modes.as_ptr(),
                    8,
                    true,
                    residual.as_raw_ptr(),
                )
            });
            assert_eq!(eager.0, vec![m, 2048]);
            assert_eq!(eager.1, DType::BFloat16);
            assert!(
                eager.2[live as usize * 2048..].iter().all(|&v| v == 0),
                "dead Hidden suffix must be initialized"
            );
            if m != 8 {
                let native = read_output("native FFN fallback reference", unsafe {
                    mlx_sys::mlx_splash_moe_with_policy(
                        handles.as_ptr(),
                        handles.len(),
                        modes.as_ptr(),
                        8,
                        true,
                        false,
                    )
                });
                let expected: Vec<u32> = native
                    .2
                    .iter()
                    .zip(&before)
                    .enumerate()
                    .map(|(i, (&ffn, &h))| {
                        if i >= live as usize * 2048 {
                            0
                        } else {
                            u32::from(
                                half::bf16::from_f32(
                                    h + half::bf16::from_bits(ffn as u16).to_f32(),
                                )
                                .to_bits(),
                            )
                        }
                    })
                    .collect();
                assert_eq!(
                    eager.2, expected,
                    "M{m} retains native FFN rounding then BF16 residual add"
                );
            }
            arrays.push(residual.clone());
            let inputs: Vec<_> = arrays.iter().map(|x| x.as_raw_ptr().cast_const()).collect();
            let mut output = std::ptr::null_mut();
            assert!(unsafe {
                mlx_sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build_residual),
                    (&mut builder as *mut ResidualBuilder).cast(),
                    inputs.as_ptr(),
                    inputs.len(),
                    &mut output,
                    1,
                    true,
                )
            });
            let got = read_output("actual source residual compiled", output);
            assert_eq!(got, eager, "budget={budget} M={m}");
            assert_eq!(builder.traces.load(Ordering::Relaxed), 1);
            assert_eq!(
                residual.to_float32().unwrap().as_ref(),
                before.as_slice(),
                "aliased residual input remains read-only"
            );
        }
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}

#[test]
#[ignore = "exclusive GPU, actual M8 weights and independent Splash budget1/3/8 output oracles"]
fn source_math_actual_budget_matches_source_replay() {
    assert_eq!(
        std::env::var("MLX_SPLASH_MOE_SOURCE_MATH").as_deref(),
        Ok("1")
    );
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_BUDGET_ORACLE_DIR").is_some());
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_CHECK_REPLAY").is_some());
    grouped_actual_layer_stage_comparison();
}

fn source_budget_replay(saved: &[MxArray], modes: &[u32]) {
    let directory =
        std::path::PathBuf::from(std::env::var("SPLASH_MOE_SOURCE_BUDGET_ORACLE_DIR").unwrap());
    let full = std::path::PathBuf::from(std::env::var("SPLASH_MOE_SOURCE_ORACLE_DIR").unwrap());
    let snapshot = std::env::var("SPLASH_MOE_INPUT").unwrap();
    let layer: u64 = std::env::var("SPLASH_MOE_LAYER").unwrap().parse().unwrap();
    let mut builder = Builder {
        policy: Some(true),
        modes: modes.to_vec(),
        traces: AtomicUsize::new(0),
    };
    let key = 0x7372634275643800;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    let mut full_values = Vec::new();
    for live in [8usize, 1, 3, 8, 3, 1] {
        let oracle = if live == 8 {
            full.clone()
        } else {
            directory.join(format!("budget{live}"))
        };
        let run: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("run.json")).unwrap()).unwrap();
        let routes: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("routes.json")).unwrap()).unwrap();
        assert_eq!(run["snapshot"].as_str(), Some(snapshot.as_str()));
        assert_eq!(run["layer"].as_u64(), Some(layer));
        assert_eq!(run["routing_mode"], "native");
        assert_eq!(run["zero_residual"], true);
        assert_eq!(run["router_bf16"], true);
        assert_eq!(run["shared_bf16"], true);
        assert_eq!(routes["live_rows"].as_u64(), Some(live as u64));
        let mut a = saved.to_vec();
        a.push(MxArray::from_int32(&[live as i32], &[]).unwrap());
        let handles: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
        // Explicit loaded policy wins even while the process environment is OFF.
        set_flag(false);
        let eager = read_output("source budget eager", unsafe {
            mlx_sys::mlx_splash_moe_with_policy(
                handles.as_ptr(),
                handles.len(),
                modes.as_ptr(),
                8,
                true,
                true,
            )
        });
        assert_eq!(eager.0, vec![8, 2048]);
        assert_eq!(eager.1, DType::BFloat16);
        require_source_bits(&oracle, "output.bf16", &eager.2);
        assert!(
            eager.2[live * 2048..].iter().all(|&bits| bits == 0),
            "dead suffix must be exact positive zero"
        );
        assert!(
            eager
                .2
                .iter()
                .all(|&bits| f32::from_bits(bits << 16).is_finite())
        );
        let values: Vec<f32> = eager
            .2
            .iter()
            .map(|&bits| f32::from_bits(bits << 16))
            .collect();
        if full_values.is_empty() {
            full_values = values.clone();
        }
        let peak = full_values[..live * 2048]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let max = values[..live * 2048]
            .iter()
            .zip(&full_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "SOURCE_BUDGET live={live} full8_prefix_peak_error={} independent_source_bits=exact",
            max / peak.max(1e-20)
        );
        let const_handles: Vec<_> = handles.iter().map(|h| h.cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut builder as *mut Builder).cast(),
                const_handles.as_ptr(),
                const_handles.len(),
                &mut out,
                1,
                true,
            )
        });
        start_counting();
        let compiled = read_output("source budget compiled", out);
        assert_eq!(family_count("splash_moe_source_math"), 1);
        assert_eq!(family_count("splash_moe_source_column"), 1);
        stop_counting();
        assert_eq!(compiled, eager);
        assert_eq!(builder.traces.load(Ordering::Relaxed), 1);
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}

#[test]
#[ignore = "exclusive GPU, original Splash production combine with nonzero captured BF16 residual"]
fn source_math_actual_residual_matches_source_combine() {
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_RESIDUAL_ORACLE_DIR").is_some());
    assert!(std::env::var_os("SPLASH_MOE_SOURCE_CHECK_REPLAY").is_some());
    assert_eq!(
        std::env::var("MLX_SPLASH_MOE_SOURCE_MATH").as_deref(),
        Ok("1")
    );
    grouped_actual_layer_stage_comparison();
}

fn source_residual_oracle_replay(saved: &[MxArray], modes: &[u32]) {
    let directory =
        std::path::PathBuf::from(std::env::var("SPLASH_MOE_SOURCE_RESIDUAL_ORACLE_DIR").unwrap());
    let snapshot = std::env::var("SPLASH_MOE_INPUT").unwrap();
    let layer: u64 = std::env::var("SPLASH_MOE_LAYER").unwrap().parse().unwrap();
    let residual = saved[0].clone();
    let before = residual.to_float32().unwrap().to_vec();
    let residual_bits: Vec<u32> = before
        .iter()
        .map(|&v| u32::from(half::bf16::from_f32(v).to_bits()))
        .collect();
    let mut builder = ResidualBuilder {
        budget: true,
        modes: modes.to_vec(),
        traces: AtomicUsize::new(0),
    };
    let key = 0x7372635265734f00;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    for live in [8usize, 1, 3, 8, 3, 1] {
        let oracle = directory.join(format!("budget{live}"));
        let run: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("run.json")).unwrap()).unwrap();
        assert_eq!(run["snapshot"].as_str(), Some(snapshot.as_str()));
        assert_eq!(run["layer"].as_u64(), Some(layer));
        assert_eq!(run["routing_mode"], "native");
        assert_eq!(run["residual_mode"], "captured_x_bf16");
        assert_eq!(run["combine_pipeline"], "moe_combine");
        assert_eq!(run["live_rows"].as_u64(), Some(live as u64));
        require_source_bits(&oracle, "residual.bf16", &residual_bits);
        let mut arrays = saved.to_vec();
        arrays.push(MxArray::from_int32(&[live as i32], &[]).unwrap());
        let handles: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
        set_flag(false);
        let eager = read_output("source residual original-combine oracle", unsafe {
            mlx_sys::mlx_splash_moe_with_residual(
                handles.as_ptr(),
                handles.len(),
                modes.as_ptr(),
                8,
                true,
                residual.as_raw_ptr(),
            )
        });
        assert_eq!(eager.0, vec![8, 2048]);
        assert_eq!(eager.1, DType::BFloat16);
        require_source_bits(&oracle, "output-live.bf16", &eager.2[..live * 2048]);
        assert!(
            eager.2[live * 2048..].iter().all(|&v| v == 0),
            "native masked suffix stays zero"
        );
        if live == 8 {
            let plain: Vec<_> = saved.iter().map(MxArray::as_raw_ptr).collect();
            let no_budget = read_output("source residual without budget", unsafe {
                mlx_sys::mlx_splash_moe_with_residual(
                    plain.as_ptr(),
                    plain.len(),
                    modes.as_ptr(),
                    8,
                    true,
                    residual.as_raw_ptr(),
                )
            });
            assert_eq!(no_budget, eager);
        }
        arrays.push(residual.clone());
        let inputs: Vec<_> = arrays.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build_residual),
                (&mut builder as *mut ResidualBuilder).cast(),
                inputs.as_ptr(),
                inputs.len(),
                &mut out,
                1,
                true,
            )
        });
        let compiled = read_output("compiled source residual original-combine oracle", out);
        assert_eq!(compiled, eager);
        assert_eq!(builder.traces.load(Ordering::Relaxed), 1);
        assert_eq!(residual.to_float32().unwrap().as_ref(), before.as_slice());
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}
