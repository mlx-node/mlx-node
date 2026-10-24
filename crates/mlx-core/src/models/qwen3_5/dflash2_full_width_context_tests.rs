// Included in dflash2::tests to reuse its small deterministic model fixtures.
fn full_width_test_model() -> super::DFlash2Model {
    let mut model = three_layer_tiny_draft();
    model.config.block_size = 7;
    model.config.sliding_window = 8;
    model.selector.top_k = 16;
    for layer in &mut model.layers {
        layer.attention.sliding_window = 8;
    }
    model
}

fn full_width_test_taps(model: &super::DFlash2Model, seed: u64) -> Vec<MxArray> {
    unsafe { mlx_sys::mlx_seed(seed) };
    model
        .config
        .target_layers
        .iter()
        .map(|_| {
            MxArray::random_normal(
                &[1, 8, model.config.hidden_size as i64],
                0.0,
                0.5,
                Some(DType::BFloat16),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn full_width_context_retained_bits_match_full_m8_projection() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let mut unmerged = full_width_test_model();
    unmerged.config.sliding_window = 9;
    let quantized_unmerged = quantized_draft_model(1, 256);
    assert!(quantized_unmerged.context_kv.is_none());
    let mut merged = quantized_draft_model(2, 256);
    merged.merge_projections().unwrap();
    assert!(merged.context_kv.is_some(), "cross-layer k|v merge");
    for (branch, model, base) in [
        ("unmerged", &unmerged, 0),
        ("quantized unmerged", &quantized_unmerged, 0),
        ("merged", &merged, 0),
        ("merged", &merged, 5),
    ] {
        let fused = model
            .fuse_context(&full_width_test_taps(model, 0x8FBF))
            .unwrap();
        let full = model.project_context(&fused, base).unwrap();
        for keep in 1..=8usize {
            let ids = (0..keep).map(|i| i as u32).collect::<Vec<_>>();
            let mut context = super::DFlash2ContextCache::new(&model.config);
            let mut expected_ids = vec![0; base as usize];
            context.record_only(&expected_ids).unwrap();
            context
                .append_full_width_prefix(model, &fused, base, &ids)
                .unwrap();
            assert_eq!(context.logical_len, base + keep as i32);
            expected_ids.extend_from_slice(&ids);
            assert_eq!(context.token_history, expected_ids);
            for (window, (full_keys, full_values)) in context.layers.iter().zip(&full) {
                let (keys, values) = window.view().unwrap();
                assert_eq!(
                    bits(&keys),
                    bits(&full_keys.slice_axis(2, 0, keep as i64).unwrap()),
                    "{branch} K base={base} keep={keep}"
                );
                assert_eq!(
                    bits(&values),
                    bits(&full_values.slice_axis(2, 0, keep as i64).unwrap()),
                    "{branch} V base={base} keep={keep}"
                );
            }
        }
    }
}

#[test]
fn full_width_context_stores_only_retained_prefix_through_compaction() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let mut model = full_width_test_model();
    model.config.sliding_window = 17;
    for layer in &mut model.layers {
        layer.attention.sliding_window = 17;
    }
    let window_rows = 16i64;
    let mut context = super::DFlash2ContextCache::new(&model.config);
    let mut expected: Vec<Option<(MxArray, MxArray)>> =
        (0..model.layers.len()).map(|_| None).collect();
    let mut tokens = Vec::new();
    let mut saw_compaction = false;
    let mut previous_end = 0;
    for cycle in 0..80 {
        let keep = [1usize, 3, 8, 2, 7][cycle % 5];
        let taps = full_width_test_taps(&model, 0x8FC0 + cycle as u64);
        let fused = model.fuse_context(&taps).unwrap();
        let projected = model.project_context(&fused, context.logical_len).unwrap();
        for (reference, (keys, values)) in expected.iter_mut().zip(projected) {
            let keys = keys.slice_axis(2, 0, keep as i64).unwrap();
            let values = values.slice_axis(2, 0, keep as i64).unwrap();
            let (keys, values) = match reference.take() {
                Some((old_k, old_v)) => (
                    MxArray::concatenate(&old_k, &keys, 2).unwrap(),
                    MxArray::concatenate(&old_v, &values, 2).unwrap(),
                ),
                None => (keys, values),
            };
            let rows = keys.shape_at(2).unwrap();
            *reference = Some((
                keys.slice_axis(2, (rows - window_rows).max(0), rows)
                    .unwrap(),
                values
                    .slice_axis(2, (rows - window_rows).max(0), rows)
                    .unwrap(),
            ));
        }
        let retained: Vec<_> = (0..keep).map(|i| ((cycle + i) % 32) as u32).collect();
        context
            .append_full_width_prefix(&model, &fused, context.logical_len, &retained)
            .unwrap();
        context.eval().unwrap();
        tokens.extend_from_slice(&retained);
        assert_eq!(context.logical_len as usize, tokens.len());
        assert_eq!(context.token_history, tokens);
        for (window, reference) in context.layers.iter().zip(&expected) {
            let (keys, values) = window.view().unwrap();
            let (want_k, want_v) = reference.as_ref().unwrap();
            assert_eq!(bits(&keys), bits(want_k), "K cycle {cycle}");
            assert_eq!(bits(&values), bits(want_v), "V cycle {cycle}");
            assert_eq!(
                window.len() as usize,
                tokens.len().min(window_rows as usize)
            );
        }
        let end = context.layers[0].end;
        saw_compaction |= end < previous_end;
        previous_end = end;
    }
    assert!(saw_compaction, "fixture must exhaust reserved capacity");
}

#[test]
fn full_width_context_preflight_errors_do_not_change_cache_or_provenance() {
    let model = full_width_test_model();
    let fused = model
        .fuse_context(&full_width_test_taps(&model, 0x8FC1))
        .unwrap();
    let mut context = super::DFlash2ContextCache::new(&model.config);
    context
        .append_full_width_prefix(&model, &fused, 0, &[1, 2, 3])
        .unwrap();
    context.eval().unwrap();
    let before = context
        .layers
        .iter()
        .map(|w| {
            (
                bits(w.keys.as_ref().unwrap()),
                bits(w.values.as_ref().unwrap()),
                w.start,
                w.end,
            )
        })
        .collect::<Vec<_>>();
    for (input, base, ids) in [
        (fused.clone(), 2, vec![4]),
        (fused.clone(), 3, vec![]),
        (fused.clone(), 3, vec![4; 9]),
        (fused.slice_axis(1, 0, 7).unwrap(), 3, vec![4]),
    ] {
        assert!(
            context
                .append_full_width_prefix(&model, &input, base, &ids)
                .is_err()
        );
        assert_eq!(context.logical_len, 3);
        assert_eq!(context.token_history, [1, 2, 3]);
        for (window, (keys, values, start, end)) in context.layers.iter().zip(&before) {
            assert_eq!(bits(window.keys.as_ref().unwrap()), *keys);
            assert_eq!(bits(window.values.as_ref().unwrap()), *values);
            assert_eq!((window.start, window.end), (*start, *end));
        }
    }
    // An ordinary short tail still uses the established exact-height append.
    let tail = fused.slice_axis(1, 0, 2).unwrap();
    context.append(&model, &tail, 3, &[4, 5]).unwrap();
    assert_eq!(context.logical_len, 5);
    assert_eq!(context.token_history, [1, 2, 3, 4, 5]);
}

#[test]
fn full_width_context_commit_feeds_eager_and_compiled_draft_identically() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let model = full_width_test_model();
    let embedding = Embedding::from_weight(
        &MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap(),
    )
    .unwrap();
    let mut eager = super::DFlash2ContextCache::new(&model.config);
    let mut compiled = super::DFlash2ContextCache::new(&model.config);
    for (cycle, keep) in [1usize, 3, 8, 2, 7, 4].into_iter().enumerate() {
        let fused = model
            .fuse_context(&full_width_test_taps(&model, 0x8FC2 + cycle as u64))
            .unwrap();
        let ids: Vec<_> = (0..keep).map(|i| ((i + cycle) % 32) as u32).collect();
        eager
            .append_full_width_prefix(&model, &fused, eager.logical_len, &ids)
            .unwrap();
        compiled
            .append_full_width_prefix(&model, &fused, compiled.logical_len, &ids)
            .unwrap();
        eager.eval().unwrap();
        compiled.eval().unwrap();
        let anchor = (cycle + 9) as u32;
        let mut confidence = None;
        let (expected, _) = model
            .propose(
                &embedding,
                None,
                &mut eager,
                anchor,
                7,
                0.0,
                true,
                Some(&mut confidence),
                &mut rand::rng(),
            )
            .unwrap();
        let super::SelectorPath::Device(expected) = expected else {
            panic!("host path")
        };
        MxArray::eval_arrays(&[&expected]).unwrap();
        let mut block = vec![0i32; 8];
        block[0] = anchor as i32;
        let block = MxArray::from_int32(&block, &[1, 8]).unwrap();
        let (actual, probability) = model
            .propose_compiled(&embedding, None, &mut compiled, &block, true)
            .unwrap()
            .unwrap();
        MxArray::eval_arrays(&[&actual]).unwrap();
        assert_eq!(
            actual.to_int32().unwrap().as_ref(),
            expected.to_int32().unwrap().as_ref()
        );
        assert_eq!(
            probability.unwrap().to_float32().unwrap().as_ref(),
            confidence.unwrap().to_float32().unwrap().as_ref()
        );
        assert_eq!(eager.token_history, compiled.token_history);
        for (left, right) in eager.layers.iter().zip(&compiled.layers) {
            assert_eq!(
                bits(left.keys.as_ref().unwrap()),
                bits(right.keys.as_ref().unwrap())
            );
            assert_eq!(
                bits(left.values.as_ref().unwrap()),
                bits(right.values.as_ref().unwrap())
            );
        }
    }
    assert_eq!(
        model.compiled.traces.load(super::Ordering::Relaxed),
        1,
        "fixed full-block mask mode must reuse one graph across positions and windows"
    );
}

// Exact affine endpoints avoid using the implementation's dequantizer as the
// reference. Each 64-weight group contains all16 representable levels; its
// scale and bias are exact BF16 powers of two. Only16 output patterns repeat.
fn full_width_projection_fixture(k: usize, n: usize) -> (LinearProj, Vec<f32>, MxArray) {
    let weights: Vec<_> = (0..n * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            (((row % 16 * 3 + col) % 16) as f32 - 8.0) / 128.0
        })
        .collect();
    let weight = MxArray::from_float32(&weights, &[n as i64, k as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let projection = LinearProj::Quantized(
        super::quantize_draft_weight(&weight, "full_width_context_test").unwrap(),
    );
    let input: Vec<_> = (0..8 * k)
        .map(|i| half::bf16::from_f32(((i * 37 + i / k * 13) % 257) as f32 / 257.0 - 0.5).to_f32())
        .collect();
    let array = MxArray::from_float32(&input, &[1, 8, k as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    (projection, input, array)
}

#[test]
#[ignore = "actual FC/KV geometry and dispatch counters require the exclusive GPU slot"]
fn full_width_context_actual_affine_shapes_match_f64() {
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    for (k, n) in [(16384usize, 2048usize), (2048, 12288)] {
        let (projection, input, array) = full_width_projection_fixture(k, n);
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let output = projection.forward(&array).unwrap().to_float32().unwrap();
        let tensor = unsafe { mlx_sys::mlx_test_kquant_family_count(c"qmm_m8_nax_t64".as_ptr()) };
        let register = unsafe { mlx_sys::mlx_test_kquant_family_count(c"qmv_sg8_t64".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        assert_eq!(tensor + register, 1, "K={k} N={n} must use one M8 kernel");
        let mut expected = vec![0f64; 8 * 16];
        for row in 0..8 {
            for pattern in 0..16 {
                expected[row * 16 + pattern] = (0..k)
                    .map(|col| {
                        f64::from(input[row * k + col]) * (((pattern * 3 + col) % 16) as f64 - 8.0)
                            / 128.0
                    })
                    .sum();
            }
        }
        let peak = expected.iter().map(|v| v.abs()).fold(1e-9, f64::max);
        let mut max = 0f64;
        let mut squares = 0f64;
        for row in 0..8 {
            for col in 0..n {
                let error =
                    (f64::from(output[row * n + col]) - expected[row * 16 + col % 16]).abs();
                max = max.max(error);
                squares += error * error;
            }
        }
        let rms = (squares / (8 * n) as f64).sqrt();
        eprintln!(
            "context M8 K={k} N={n} tensor={tensor} register={register} max/peak={} rms/peak={}",
            max / peak,
            rms / peak
        );
        assert!(max <= peak * 0.005 + 1e-6 && rms <= peak * 0.0015 + 1e-6);
    }
}

#[test]
#[ignore = "actual FC+merged-KV operator timing requires the exclusive GPU slot"]
fn full_width_context_projection_microbenchmark() {
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    let (fc, _, captured) = full_width_projection_fixture(16384, 2048);
    let (kv, _, _) = full_width_projection_fixture(2048, 12288);
    let norm = crate::nn::RMSNorm::from_weight(
        &MxArray::ones(&[2048], Some(DType::BFloat16)).unwrap(),
        Some(1e-6),
    )
    .unwrap();
    let batch = |rows| {
        let input = captured.slice_axis(1, 0, rows).unwrap();
        (0..16)
            .map(|_| {
                kv.forward(&norm.forward(&fc.forward(&input).unwrap()).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    // Both full width and exact retained height use the same resident weights,
    // inputs and norm. Construct graphs before timing; each batch forces16
    // independent FC+norm+KV chains, excluding host graph-construction time.
    for rows in [8i64, 1, 2, 3, 4, 5, 6, 7, 8] {
        for _ in 0..8 {
            let out = batch(rows);
            MxArray::eval_arrays(&out.iter().collect::<Vec<_>>()).unwrap();
        }
        let mut samples = Vec::new();
        for _ in 0..31 {
            let out = batch(rows);
            let start = std::time::Instant::now();
            MxArray::eval_arrays(&out.iter().collect::<Vec<_>>()).unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1e6 / 16.0);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "context FC+norm+mergedKV rows={rows} median_us={} p10_us={}",
            samples[15], samples[3]
        );
    }
}

#[test]
fn compiled_draft_attention_masked_key_growth_and_saturation() {
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    use crate::nn::{Linear, RMSNorm, RoPE};
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.0 {
                    Some(value) => std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", value),
                    None => std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX"),
                }
            }
        }
    }
    let _restore = Restore(std::env::var_os("MLX_DFLASH2_ATTENTION_NAX"));
    let unused = || {
        LinearProj::Standard(
            Linear::from_weights(
                &MxArray::zeros(&[1, 1], Some(DType::BFloat16)).unwrap(),
                None,
            )
            .unwrap(),
        )
    };
    let norm = || {
        RMSNorm::from_weight(
            &MxArray::ones(&[128], Some(DType::BFloat16)).unwrap(),
            Some(1e-6),
        )
        .unwrap()
    };
    let mut attention = super::DFlash2Attention {
        tensor_attention_enabled: false,
        q_proj: unused(),
        k_proj: unused(),
        v_proj: unused(),
        o_proj: LinearProj::Standard(
            Linear::from_weights(
                &MxArray::random_normal(&[4, 4096], 0.0, 0.02, Some(DType::BFloat16)).unwrap(),
                None,
            )
            .unwrap(),
        ),
        qkv_proj: None,
        q_norm: norm(),
        k_norm: norm(),
        rope: RoPE::new(128, Some(false), Some(10_000_000.0), None),
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        sliding_window: 2048,
    };
    for enabled in [false, true] {
        attention.tensor_attention_enabled = enabled;
        unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", if enabled { "1" } else { "0" }) };
        let key = if enabled {
            0xDFA2_FF01_0000_0001
        } else {
            0xDFA2_FF01_0000_0000
        };
        let traces = std::cell::Cell::new(0);
        // All cases have a mask; cross both near-saturation key growth and
        // changing absolute positions at the full window without retracing.
        for (index, keys_len) in [2049i64, 2052, 2055, 2055, 2049].into_iter().enumerate() {
            let query_base = 32768 + index as i32 * 17;
            let key_base = query_base - (keys_len - 8) as i32;
            let queries =
                MxArray::random_normal(&[1, 32, 8, 128], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
            let keys =
                MxArray::random_normal(&[1, 8, keys_len, 128], 0.0, 0.5, Some(DType::BFloat16))
                    .unwrap();
            let values =
                MxArray::random_normal(&[1, 8, keys_len, 128], 0.0, 0.5, Some(DType::BFloat16))
                    .unwrap();
            let mask = super::non_causal_sliding_mask(query_base, 8, key_base, keys_len, 2048)
                .unwrap()
                .unwrap();
            let broadcast = mask.broadcast_to(&[1, 32, 8, keys_len]).unwrap();
            if enabled {
                let mut raw = std::ptr::null_mut();
                assert_eq!(
                    unsafe {
                        mlx_sys::mlx_dflash2_attention(
                            queries.as_raw_ptr(),
                            keys.as_raw_ptr(),
                            values.as_raw_ptr(),
                            mask.as_raw_ptr(),
                            1.0f32 / 128.0f32.sqrt(),
                            &mut raw,
                        )
                    },
                    1
                );
                unsafe { mlx_sys::mlx_array_delete(raw) };
            }
            let expected = attention
                .attend_masked(&queries, &keys, &values, Some(&mask))
                .unwrap();
            let expected = bits(&expected);
            let mut builder = |args: &[MxArray]| -> napi::Result<Vec<MxArray>> {
                traces.set(traces.get() + 1);
                Ok(vec![attention.attend_masked_with_broadcast(
                    &args[0],
                    &args[1],
                    &args[2],
                    Some(&args[3]),
                    Some(&args[4]),
                )?])
            };
            let actual = crate::compiled_graph::invoke_compiled_graph(
                key,
                &[&queries, &keys, &values, &mask, &broadcast],
                1,
                true,
                &mut builder,
            )
            .unwrap()
            .unwrap();
            assert_eq!(bits(&actual[0]), expected, "custom={enabled} K={keys_len}");
        }
        assert_eq!(
            traces.get(),
            1,
            "custom={enabled}: masked K growth must not retrace"
        );
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    }
}
