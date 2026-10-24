#[test]
fn draft_execution_policy_is_scoped_and_honors_explicit_overrides() {
    use super::DraftExecutionPolicy;
    let dense = DraftExecutionPolicy::from_values(None, None, None);
    let moe = dense.with_moe_profile(true);
    assert!(!dense.compile_draft() && !dense.tensor_attention());
    assert!(moe.compile_draft() && moe.tensor_attention());
    assert!(!moe.with_moe_profile(false).compile_draft());
    for (value, enabled) in [("0", false), ("1", true), ("true", false), ("", false)] {
        for supported in [false, true] {
            let policy = DraftExecutionPolicy::from_values(Some(value), Some(value), None)
                .with_moe_profile(supported);
            assert_eq!(policy.compile_draft(), enabled);
            assert_eq!(policy.tensor_attention(), enabled);
        }
    }
    let mixed = DraftExecutionPolicy::from_values(Some("0"), None, None).with_moe_profile(true);
    assert!(!mixed.compile_draft() && mixed.tensor_attention());
    let mixed = DraftExecutionPolicy::from_values(None, Some("0"), None).with_moe_profile(true);
    assert!(mixed.compile_draft() && !mixed.tensor_attention());
    // A second load's explicit settings do not change either earlier snapshot.
    let _other = DraftExecutionPolicy::from_values(Some("0"), Some("0"), None);
    assert!(moe.compile_draft() && moe.tensor_attention());
    assert!(!dense.compile_draft() && !dense.tensor_attention());
}

#[test]
fn draft_execution_defaults_require_the_measured_moe_structure() {
    let target = [2048, 40, 256, 8, 512];
    let config = DFlash2Config {
        block_size: 7,
        mask_token_id: 248077,
        target_layers: vec![1, 6, 11, 16, 22, 27, 32, 37],
        target_num_layers: 40,
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 6,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 248320,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 262144,
        sliding_window: 2048,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 256,
        selector_top_k: 16,
        rope_theta: 10000000.0,
    };
    assert!(config.supports_moe_execution_defaults(target));
    for index in 0..target.len() {
        let mut other = target;
        other[index] -= 1;
        assert!(!config.supports_moe_execution_defaults(other));
    }
    for index in 0..14 {
        let mut other = config.clone();
        match index {
            0 => other.hidden_size -= 1,
            1 => other.intermediate_size -= 1,
            2 => other.target_num_layers -= 1,
            3 => {
                other.target_layers.pop();
            }
            4 => other.num_hidden_layers -= 1,
            5 => other.num_attention_heads -= 1,
            6 => other.num_key_value_heads -= 1,
            7 => other.head_dim -= 1,
            8 => other.sliding_window -= 1,
            9 => other.block_size -= 1,
            10 => other.conv_group_size -= 1,
            11 => other.conv_kernel_size -= 1,
            12 => other.selector_rank -= 1,
            13 => other.selector_top_k -= 1,
            _ => unreachable!(),
        }
        assert!(
            !other.supports_moe_execution_defaults(target),
            "field {index}"
        );
    }
}

// Source convolution integration gates. Metal cases run individually in a
// fresh process on the measured device; they never load a target checkpoint.
fn source_conv_measured_config() -> DFlash2Config {
    DFlash2Config {
        block_size: 7,
        mask_token_id: 248077,
        target_layers: vec![1, 6, 11, 16, 22, 27, 32, 37],
        target_num_layers: 40,
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 6,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 248320,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 262144,
        sliding_window: 2048,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 256,
        selector_top_k: 16,
        rope_theta: 10_000_000.0,
    }
}

#[test]
fn source_draft_conv_policy_is_immutable_and_profile_scoped() {
    use super::DraftExecutionPolicy;
    for requested in [None, Some("0"), Some("1"), Some("true"), Some("")] {
        let captured = DraftExecutionPolicy::from_values(Some("0"), Some("0"), requested);
        for moe in [false, true] {
            for source in [false, true] {
                let policy = captured
                    .with_moe_profile(moe)
                    .with_source_conv_profile(source);
                assert_eq!(
                    policy.source_conv(),
                    requested == Some("1") && moe && source
                );
                assert!(!policy.compile_draft() && !policy.tensor_attention());
                let _other_load = DraftExecutionPolicy::from_values(None, None, Some("0"));
                assert_eq!(
                    policy.source_conv(),
                    requested == Some("1") && moe && source
                );
            }
        }
        assert!(
            !captured.source_conv(),
            "dense load must not select this experiment"
        );
    }
}

#[test]
fn source_draft_conv_profile_checks_exact_companion_and_kernel_epsilon() {
    let target = [2048, 40, 256, 8, 512];
    let config = source_conv_measured_config();
    assert!(config.supports_source_conv(target));
    let mut imported = config.clone();
    imported.rms_norm_eps = f64::from(1e-6f32);
    assert!(imported.supports_source_conv(target));
    for i in 0..target.len() {
        let mut wrong = target;
        wrong[i] -= 1;
        assert!(!config.supports_source_conv(wrong));
    }
    for i in 0..21 {
        let mut wrong = config.clone();
        match i {
            0 => wrong.hidden_size -= 1,
            1 => wrong.intermediate_size -= 1,
            2 => wrong.target_num_layers -= 1,
            3 => wrong.target_layers[0] += 1,
            4 => wrong.num_hidden_layers -= 1,
            5 => wrong.num_attention_heads -= 1,
            6 => wrong.num_key_value_heads -= 1,
            7 => wrong.head_dim -= 1,
            8 => wrong.sliding_window -= 1,
            9 => wrong.block_size -= 1,
            10 => wrong.conv_group_size -= 1,
            11 => wrong.conv_kernel_size -= 1,
            12 => wrong.selector_rank -= 1,
            13 => wrong.selector_top_k -= 1,
            14 => wrong.mask_token_id -= 1,
            15 => wrong.vocab_size -= 1,
            16 => wrong.max_position_embeddings -= 1,
            17 => wrong.rms_norm_eps = 1e-5,
            18 => wrong.rope_theta = 10000.0,
            19 => {
                wrong.target_layers.pop();
            }
            20 => wrong.rms_norm_eps = f64::NAN,
            _ => unreachable!(),
        }
        assert!(!wrong.supports_source_conv(target), "changed field {i}");
    }
}

fn source_conv_test_linear(input: i64, output: i64) -> LinearProj {
    LinearProj::Standard(
        Linear::from_weights(
            &MxArray::random_normal(&[output, input], 0.0, 0.004, Some(DType::BFloat16)).unwrap(),
            None,
        )
        .unwrap(),
    )
}

fn source_conv_test_norm(size: i64) -> crate::nn::RMSNorm {
    crate::nn::RMSNorm::from_weight(
        &MxArray::random_normal(&[size], 1.0, 0.125, Some(DType::BFloat16)).unwrap(),
        Some(1e-6),
    )
    .unwrap()
}

fn source_conv_test_conv(enabled: bool) -> super::GroupedDynamicCausalConv {
    super::GroupedDynamicCausalConv {
        source_enabled: enabled,
        base_kernel: MxArray::random_normal(&[2, 2, 2048], 0.0, 0.25, Some(DType::BFloat16))
            .unwrap(),
        kernel_projection: source_conv_test_linear(2048, 512),
        kernel_size: 2,
        group_size: 16,
    }
}

fn source_conv_test_raw(
    conv: &super::GroupedDynamicCausalConv,
    x: &MxArray,
    dynamic: &MxArray,
    side: i32,
    residual: Option<&MxArray>,
) -> MxArray {
    MxArray::from_handle(
        unsafe {
            mlx_sys::mlx_source_dflash2_conv(
                x.as_raw_ptr(),
                dynamic.as_raw_ptr(),
                conv.base_kernel.as_raw_ptr(),
                side,
                residual.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            )
        },
        "source conv integration oracle",
    )
    .unwrap()
}

#[test]
#[ignore = "fresh-process measured Metal device; source convolution carrier gate"]
fn source_draft_conv_prepare_finish_adds_and_normalizes_once() {
    assert!(unsafe { mlx_sys::mlx_source_dflash2_conv_available() });
    unsafe {
        mlx_sys::mlx_seed(0xDC01);
    }
    let conv = source_conv_test_conv(true);
    let x = MxArray::random_normal(&[1, 8, 2048], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    let residual = MxArray::random_normal(&[1, 8, 2048], 0.0, 0.75, Some(DType::BFloat16)).unwrap();
    let norm = source_conv_test_norm(2048);
    let (prepared, dynamic) = conv.prepare(&x).unwrap();
    let expected_dynamic = conv
        .kernel_projection
        .forward(&x)
        .unwrap()
        .reshape(&[1, 8, 2, 2, 128])
        .unwrap();
    assert_eq!(bits(&dynamic), bits(&expected_dynamic));
    assert_eq!(
        bits(&prepared),
        bits(&source_conv_test_raw(&conv, &x, &dynamic, 0, None))
    );
    let expected = source_conv_test_raw(&conv, &x, &dynamic, 1, Some(&residual));
    let output = conv.finish_output(&x, &dynamic, &residual).unwrap();
    assert!(matches!(&output, super::DraftLayerOutput::Hidden(_)));
    assert!(matches!(output.as_input(), super::DraftResidual::Hidden(_)));
    let (hidden, normalized) = output.hidden_and_normalized(&norm).unwrap();
    assert_eq!(bits(&hidden), bits(&expected));
    assert_eq!(bits(&normalized), bits(&norm.forward(&expected).unwrap()));
    assert_ne!(
        bits(&hidden),
        bits(&expected.add(&residual).unwrap()),
        "fixture catches a second add"
    );
    assert_ne!(
        bits(&normalized),
        bits(&norm.forward(&normalized).unwrap()),
        "fixture catches a second norm"
    );
    // Native finish deliberately rounds the delta before the residual addition;
    // this fixture must distinguish the two arithmetic contracts.
    let native = conv.finish(&x, &dynamic).unwrap().add(&residual).unwrap();
    assert_ne!(
        bits(&hidden),
        bits(&native),
        "fixture must detect premature BF16 rounding"
    );
}

#[test]
#[ignore = "fresh-process measured Metal device; exact native tail fallback"]
fn source_draft_conv_native_tails_are_exact() {
    assert!(unsafe { mlx_sys::mlx_source_dflash2_conv_available() });
    unsafe {
        mlx_sys::mlx_seed(0xDC02);
    }
    let mut conv = source_conv_test_conv(false);
    let norm = source_conv_test_norm(2048);
    for rows in [1, 2, 7, 9] {
        let x = MxArray::random_normal(&[1, rows, 2048], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
        let residual =
            MxArray::random_normal(&[1, rows, 2048], 0.0, 0.75, Some(DType::BFloat16)).unwrap();
        conv.source_enabled = false;
        let (expected_x, expected_dynamic) = conv.prepare(&x).unwrap();
        let expected = conv
            .finish_output(&x, &expected_dynamic, &residual)
            .unwrap();
        let (expected_hidden, expected_norm) = expected.hidden_and_normalized(&norm).unwrap();
        let expected_bits = (
            bits(&expected_x),
            bits(&expected_dynamic),
            bits(&expected_hidden),
            bits(&expected_norm),
        );
        conv.source_enabled = true;
        let (actual_x, actual_dynamic) = conv.prepare(&x).unwrap();
        let actual = conv.finish_output(&x, &actual_dynamic, &residual).unwrap();
        assert!(
            matches!(&actual, super::DraftLayerOutput::Pending(_, _)),
            "rows={rows}"
        );
        let (actual_hidden, actual_norm) = actual.hidden_and_normalized(&norm).unwrap();
        assert_eq!(
            expected_bits,
            (
                bits(&actual_x),
                bits(&actual_dynamic),
                bits(&actual_hidden),
                bits(&actual_norm)
            ),
            "rows={rows}"
        );
    }
}

// Actual per-layer attention/MLP/conv geometry; one layer and a small selector
// vocabulary bound memory. Production six-layer/profile selection is tested
// separately above and by the full-model acceptance runner.
fn source_conv_test_model() -> super::DFlash2Model {
    use crate::nn::RoPE;
    let mut config = source_conv_measured_config();
    config.num_hidden_layers = 1;
    config.vocab_size = 32;
    config.mask_token_id = 0;
    config.selector_rank = 8;
    config.target_layers = vec![1];
    let layer = super::DFlash2Layer {
        attention: super::DFlash2Attention {
            tensor_attention_enabled: true,
            q_proj: source_conv_test_linear(2048, 4096),
            k_proj: source_conv_test_linear(2048, 1024),
            v_proj: source_conv_test_linear(2048, 1024),
            o_proj: source_conv_test_linear(4096, 2048),
            qkv_proj: None,
            q_norm: source_conv_test_norm(128),
            k_norm: source_conv_test_norm(128),
            rope: RoPE::new(128, Some(false), Some(config.rope_theta), None),
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            sliding_window: 2048,
        },
        mlp: super::DFlash2Mlp {
            gate_proj: source_conv_test_linear(2048, 6144),
            up_proj: source_conv_test_linear(2048, 6144),
            down_proj: source_conv_test_linear(6144, 2048),
            gate_up: None,
        },
        input_norm: source_conv_test_norm(2048),
        post_attention_norm: source_conv_test_norm(2048),
        attention_conv: source_conv_test_conv(true),
        mlp_conv: source_conv_test_conv(true),
    };
    super::DFlash2Model {
        // Eager control deliberately disables the proposal compile wrapper;
        // the explicit compiled method below exercises the same layer graph.
        execution_policy: super::DraftExecutionPolicy::from_values(Some("0"), Some("1"), Some("1"))
            .with_moe_profile(true)
            .with_source_conv_profile(true),
        fc: source_conv_test_linear(2048, 2048),
        hidden_norm: source_conv_test_norm(2048),
        norm: source_conv_test_norm(2048),
        selector: test_selector(32, 16, 8, 2048),
        layers: vec![layer],
        config,
        weight_bytes: 0,
        context_kv: None,
        compiled: super::CompiledDraft::default(),
    }
}

fn source_conv_test_caches(
    config: &DFlash2Config,
    live: i64,
    start: i64,
    position: i32,
) -> (super::DFlash2ContextCache, super::DFlash2ContextCache) {
    let mut eager = super::DFlash2ContextCache::new(config);
    let mut compiled = super::DFlash2ContextCache::new(config);
    eager.logical_len = position;
    compiled.logical_len = position;
    for (left, right) in eager.layers.iter_mut().zip(&mut compiled.layers) {
        for key in [true, false] {
            let data = MxArray::random_normal(
                &[1, 8, start + live + 16, 128],
                0.0,
                0.25,
                Some(DType::BFloat16),
            )
            .unwrap();
            let copy = data.deep_copy().unwrap();
            MxArray::eval_arrays(&[&data, &copy]).unwrap();
            if key {
                left.keys = Some(data);
                right.keys = Some(copy);
            } else {
                left.values = Some(data);
                right.values = Some(copy);
            }
        }
        left.start = start;
        left.end = start + live;
        right.start = start;
        right.end = start + live;
    }
    (eager, compiled)
}

#[test]
#[ignore = "fresh-process measured Metal device; bounded one-layer proposal replay and failure recovery"]
fn source_draft_conv_compiled_replay_preserves_window_and_failure_state() {
    assert!(unsafe { mlx_sys::mlx_source_dflash2_conv_available() });
    unsafe {
        mlx_sys::mlx_seed(0xDC03);
    }
    let mut model = source_conv_test_model();
    let embedding = Embedding::from_weight(
        &MxArray::random_normal(&[32, 2048], 0.0, 0.2, Some(DType::BFloat16)).unwrap(),
    )
    .unwrap();
    for (case, (live, start, position)) in
        [(3, 0, 3), (7, 2, 100), (2047, 0, 32768), (2047, 3, 33001)]
            .into_iter()
            .enumerate()
    {
        let (mut eager, mut compiled) =
            source_conv_test_caches(&model.config, live, start, position);
        let original_k = bits(&compiled.layers[0].view().unwrap().0);
        let original_v = bits(&compiled.layers[0].view().unwrap().1);
        let mut confidence = None;
        let anchor = case as u32 + 1;
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
            panic!("host selector")
        };
        MxArray::eval_arrays(&[&expected]).unwrap();
        let block = MxArray::from_int32(&[anchor as i32, 0, 0, 0, 0, 0, 0, 0], &[1, 8]).unwrap();
        let (actual, actual_confidence) = model
            .propose_compiled(&embedding, None, &mut compiled, &block, true)
            .unwrap()
            .unwrap();
        MxArray::eval_arrays(&[&actual]).unwrap();
        assert_eq!(
            actual.to_int32().unwrap().as_ref(),
            expected.to_int32().unwrap().as_ref(),
            "path case{case}"
        );
        assert_eq!(
            actual_confidence.unwrap().to_float32().unwrap().as_ref(),
            confidence.unwrap().to_float32().unwrap().as_ref(),
            "confidence case{case}"
        );
        for (left, right) in eager.layers.iter().zip(&compiled.layers) {
            assert_eq!(
                bits(left.keys.as_ref().unwrap()),
                bits(right.keys.as_ref().unwrap()),
                "K case{case}"
            );
            assert_eq!(
                bits(left.values.as_ref().unwrap()),
                bits(right.values.as_ref().unwrap()),
                "V case{case}"
            );
            assert_eq!((right.start, right.end), (start, start + live));
        }
        assert_eq!(compiled.logical_len, position);
        assert_eq!(
            bits(&compiled.layers[0].view().unwrap().0),
            original_k,
            "proposal must not commit context"
        );
        assert_eq!(bits(&compiled.layers[0].view().unwrap().1), original_v);
    }
    assert_eq!(
        model.compiled.traces.load(super::Ordering::Relaxed),
        2,
        "one unmasked and one masked graph despite offset/window changes"
    );
    // Invalid source base is accepted by the native elementwise fallback, but
    // once source is selected its factory error must propagate before cache writes.
    model.invalidate_compiled();
    let good = model.layers[0].attention_conv.base_kernel.clone();
    model.layers[0].attention_conv.base_kernel = good.astype(DType::Float32).unwrap();
    let (mut eager, mut compiled) = source_conv_test_caches(&model.config, 3, 0, 3);
    let before = (
        bits(compiled.layers[0].keys.as_ref().unwrap()),
        bits(compiled.layers[0].values.as_ref().unwrap()),
    );
    let block = MxArray::from_int32(&[1, 0, 0, 0, 0, 0, 0, 0], &[1, 8]).unwrap();
    assert!(
        model
            .propose_compiled(&embedding, None, &mut compiled, &block, true)
            .is_err()
    );
    assert!(
        model
            .forward_hidden(&embedding, &block, 3, &mut eager)
            .is_err()
    );
    assert_eq!(
        (
            compiled.logical_len,
            compiled.layers[0].start,
            compiled.layers[0].end
        ),
        (3, 0, 3)
    );
    assert_eq!(
        before,
        (
            bits(compiled.layers[0].keys.as_ref().unwrap()),
            bits(compiled.layers[0].values.as_ref().unwrap())
        )
    );
    assert_eq!(
        before,
        (
            bits(eager.layers[0].keys.as_ref().unwrap()),
            bits(eager.layers[0].values.as_ref().unwrap())
        )
    );
    model.layers[0].attention_conv.base_kernel = good;
    model.invalidate_compiled();
    let expected = model
        .forward_hidden(&embedding, &block, 3, &mut eager)
        .unwrap();
    MxArray::eval_arrays(&[&expected]).unwrap();
    let (retry, _) = model
        .propose_compiled(&embedding, None, &mut compiled, &block, true)
        .unwrap()
        .unwrap();
    MxArray::eval_arrays(&[&retry]).unwrap();
    assert_eq!(
        bits(eager.layers[0].keys.as_ref().unwrap()),
        bits(compiled.layers[0].keys.as_ref().unwrap())
    );
    assert_eq!(
        bits(eager.layers[0].values.as_ref().unwrap()),
        bits(compiled.layers[0].values.as_ref().unwrap())
    );
    assert!(!model.compiled.disabled.load(super::Ordering::Relaxed));
    // The production wrapper latches a selected compiled error and keeps that
    // latch through turn invalidation. A corrected later turn succeeds eager.
    model.invalidate_compiled();
    let good = model.layers[0].attention_conv.base_kernel.clone();
    model.layers[0].attention_conv.base_kernel = good.astype(DType::Float32).unwrap();
    model.execution_policy.compile_override = Some(true);
    model
        .compiled
        .prepared
        .store(true, super::Ordering::Relaxed);
    let (mut retry_expected, mut recovered) = source_conv_test_caches(&model.config, 3, 0, 3);
    let before = (
        bits(recovered.layers[0].keys.as_ref().unwrap()),
        bits(recovered.layers[0].values.as_ref().unwrap()),
    );
    assert!(
        model
            .propose(
                &embedding,
                None,
                &mut recovered,
                1,
                7,
                0.0,
                true,
                None,
                &mut rand::rng()
            )
            .is_err()
    );
    assert!(model.compiled.disabled.load(super::Ordering::Relaxed));
    assert_eq!(
        before,
        (
            bits(recovered.layers[0].keys.as_ref().unwrap()),
            bits(recovered.layers[0].values.as_ref().unwrap())
        )
    );
    assert_eq!(
        (
            recovered.logical_len,
            recovered.layers[0].start,
            recovered.layers[0].end
        ),
        (3, 0, 3)
    );
    model.layers[0].attention_conv.base_kernel = good;
    model.invalidate_compiled();
    assert!(model.compiled.disabled.load(super::Ordering::Relaxed));
    let expected = model
        .forward_hidden(&embedding, &block, 3, &mut retry_expected)
        .unwrap();
    MxArray::eval_arrays(&[&expected]).unwrap();
    let (path, _) = model
        .propose(
            &embedding,
            None,
            &mut recovered,
            1,
            7,
            0.0,
            true,
            None,
            &mut rand::rng(),
        )
        .unwrap();
    let super::SelectorPath::Device(path) = path else {
        panic!("host selector")
    };
    MxArray::eval_arrays(&[&path]).unwrap();
    assert_eq!(
        bits(recovered.layers[0].keys.as_ref().unwrap()),
        bits(retry_expected.layers[0].keys.as_ref().unwrap())
    );
    assert_eq!(
        bits(recovered.layers[0].values.as_ref().unwrap()),
        bits(retry_expected.layers[0].values.as_ref().unwrap())
    );
}
